"""Vacuum Langevin MD of alanine dipeptide (OpenMM, AMBER via openmmtools) with saved backbone torsions."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

PHI_ATOMS = (4, 6, 8, 14)    # C-N-CA-C
PSI_ATOMS = (6, 8, 14, 16)   # N-CA-C-N
THETA_ATOMS = (1, 4, 6, 8)   # O-C-N-CA, first peptide bond


def dihedrals(dcd, topology):
    """(phi, psi, theta) per saved frame, via MDTraj."""
    import mdtraj as md
    traj = md.load(str(dcd), top=md.Topology.from_openmm(topology))
    return [md.compute_dihedrals(traj, [ids])[:, 0].astype(np.float64) for ids in (PHI_ATOMS, PSI_ATOMS, THETA_ATOMS)]


def _write(path, series, cfg, temperature, steps, complete):
    """Atomic torsion-series write."""
    dt_ps = cfg['save_every']*cfg['timestep_fs']*1e-3
    phi, psi, theta = (np.concatenate(s) for s in series)
    fd, tmp = tempfile.mkstemp(prefix='.alanine_', suffix='.npz', dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(tmp, phi=phi, psi=psi, theta=theta, t_ps=np.arange(len(phi), dtype=np.float64)*dt_ps,
                            dt_ps=np.array([dt_ps]), temperature_k=np.array([float(temperature)]),
                            friction_ps=np.array([float(cfg['friction_ps'])]), timestep_fs=np.array([float(cfg['timestep_fs'])]),
                            total_steps=np.array([cfg['total_steps']], dtype=np.int64), save_every=np.array([cfg['save_every']], dtype=np.int64),
                            steps_completed=np.array([steps], dtype=np.int64), is_complete=np.array([int(complete)], dtype=np.int64))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def simulate(cfg, temperature, seed, path):
    """One replica: minimize, Maxwell velocities, LangevinMiddle steps; torsions every `save_every` steps.

    Torsions and an OpenMM checkpoint are written every `checkpoint_every` steps; a rerun resumes exactly.
    """
    from openmm import LangevinMiddleIntegrator, Platform, app, unit
    from openmmtools.testsystems import AlanineDipeptideVacuum
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state, done, series = path.with_suffix('.chk'), 0, [[], [], []]
    if path.exists():
        with np.load(path) as d:
            if int(d['is_complete'][0]):
                return
            done, series = int(d['steps_completed'][0]), [[d[k]] for k in ('phi', 'psi', 'theta')]
    molecule = AlanineDipeptideVacuum()
    integrator = LangevinMiddleIntegrator(temperature*unit.kelvin, cfg['friction_ps']/unit.picosecond, cfg['timestep_fs']*unit.femtosecond)
    integrator.setRandomNumberSeed(seed)
    platform = Platform.getPlatformByName(cfg['platform'])
    properties = {'Precision': cfg['precision']} if cfg['platform'] in ('CUDA', 'OpenCL') else {}
    simulation = app.Simulation(molecule.topology, molecule.system, integrator, platform, properties)
    if done:
        simulation.loadCheckpoint(str(state))
    else:
        simulation.context.setPositions(molecule.positions)
        simulation.minimizeEnergy()
        simulation.context.setVelocitiesToTemperature(temperature*unit.kelvin, seed)
    chunk = max(cfg['checkpoint_every'], cfg['save_every'])//cfg['save_every']*cfg['save_every']
    while done < cfg['total_steps']:
        steps = min(chunk, cfg['total_steps']-done)
        dcd = path.with_name(f'.{path.stem}_{done}.dcd')
        simulation.reporters = [app.DCDReporter(str(dcd), cfg['save_every'])]
        simulation.step(steps)
        simulation.reporters = []
        for s, angles in zip(series, dihedrals(dcd, molecule.topology)):
            s.append(angles)
        dcd.unlink()
        done += steps
        simulation.saveCheckpoint(str(state))
        _write(path, series, cfg, temperature, done, done >= cfg['total_steps'])
