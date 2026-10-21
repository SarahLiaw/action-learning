"""Restrained AMBER realizations of selected angle paths: geometry and potential-energy checks."""
from __future__ import annotations

import hashlib

import numpy as np

from .evaluate import dense_path, occupancy_cost, winding
from .simulate import PHI_ATOMS, PSI_ATOMS, THETA_ATOMS

TORSIONS = (PHI_ATOMS, PSI_ATOMS)
OMEGAS = (THETA_ATOMS, (8, 14, 16, 18))
BONDI = {'H': .120, 'C': .170, 'N': .155, 'O': .152}


def periodic(a):
    return (a+np.pi) % (2*np.pi)-np.pi


def dihedral(x, ids):
    p = x[list(ids)]
    b0, b1, b2 = p[0]-p[1], p[2]-p[1], p[3]-p[2]
    b1 = b1/np.linalg.norm(b1)
    v, w = b0-np.dot(b0, b1)*b1, b2-np.dot(b2, b1)*b1
    return np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))


def rotating_atoms(topology, ids):
    """Atoms on the far side of the torsion's central bond."""
    b, c = ids[1:3]
    edges = {a.index: set() for a in topology.atoms()}
    for x, y in topology.bonds():
        edges[x.index].add(y.index)
        edges[y.index].add(x.index)
    edges[b].remove(c)
    edges[c].remove(b)
    found, front = {c}, [c]
    while front:
        x = front.pop()
        for y in edges[x]-found:
            found.add(y)
            front.append(y)
    if b in found:
        raise ValueError('torsion bond lies in a cycle')
    return sorted(found)


def set_torsion(x, ids, target, subset):
    """Rigid rotation of `subset` about the central bond to the target torsion."""
    current = dihedral(x, ids)
    angle = periodic(target-current)
    origin = x[ids[1]]
    axis = (x[ids[2]]-origin)/np.linalg.norm(x[ids[2]]-origin)
    v = x[subset]-origin
    candidates = []
    for sign in (1, -1):
        a = sign*angle
        rot = v*np.cos(a)+np.cross(axis, v)*np.sin(a)+(v@axis)[:, None]*axis[None]*(1-np.cos(a))
        y = x.copy()
        y[subset] = rot+origin
        candidates.append(y)
    return min(candidates, key=lambda y: abs(periodic(dihedral(y, ids)-target)))


def chirality(x):
    """Signed volume at C-alpha (N, C, CB)."""
    return float(np.dot(np.cross(x[6]-x[8], x[14]-x[8]), x[10]-x[8]))


class Realizer:
    """Vacuum AMBER alanine dipeptide with a stiff (phi, psi) restraint; energies exclude the restraint."""

    def __init__(self, cfg):
        import mdtraj as md
        from openmm import Context, CustomTorsionForce, LocalEnergyMinimizer, Platform, VerletIntegrator, XmlSerializer, unit
        from openmmtools.testsystems import AlanineDipeptideVacuum
        self.cfg, self.unit, self.minimizer = cfg, unit, LocalEnergyMinimizer
        molecule = AlanineDipeptideVacuum()
        self.topology, system = molecule.topology, molecule.system
        self.system_sha = hashlib.sha256(XmlSerializer.serialize(system).encode()).hexdigest()
        for force in system.getForces():
            force.setForceGroup(0)
        self.restraint = CustomTorsionForce('torsion_k*(1-cos(theta-target))')
        self.restraint.addGlobalParameter('torsion_k', cfg['torsion_k'])
        self.restraint.addPerTorsionParameter('target')
        for ids in TORSIONS:
            self.restraint.addTorsion(*ids, [0.])
        self.restraint.setForceGroup(31)
        system.addForce(self.restraint)
        self.integrator = VerletIntegrator(.001*unit.picoseconds)
        self.context = Context(system, self.integrator, Platform.getPlatformByName('Reference'))
        self.context.setPositions(np.array(molecule.positions.value_in_unit(unit.nanometer))*unit.nanometer)
        self.context.setParameter('torsion_k', 0)
        LocalEnergyMinimizer.minimize(self.context, cfg['base_tolerance'], cfg['base_iterations'])
        self.base = np.asarray(self.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        self.context.setParameter('torsion_k', cfg['torsion_k'])
        self.subsets = [rotating_atoms(self.topology, t) for t in TORSIONS]
        self.reference_chirality = np.sign(chirality(self.base))
        self.bonds = [(a.index, b.index) for a, b in self.topology.bonds()]
        self.base_lengths = np.array([np.linalg.norm(self.base[a]-self.base[b]) for a, b in self.bonds])
        self.elements = [a.element.symbol for a in self.topology.atoms()]
        mdtraj_angles = md.compute_dihedrals(md.Trajectory(self.base[None], md.Topology.from_openmm(self.topology)), np.array(TORSIONS))[0]
        assert np.max(abs(periodic(mdtraj_angles-np.array([dihedral(self.base, t) for t in TORSIONS])))) < 1e-5
        self.roundtrip = []
        for target in (np.array([-.9, 1.1]), np.array([2.9, -2.8]), np.array([.6, -.8])):
            y = self.base.copy()
            for t, value, sub in zip(TORSIONS, target, self.subsets):
                y = set_torsion(y, t, value, sub)
            error = np.max(abs(periodic(np.array([dihedral(y, t) for t in TORSIONS])-target)))
            assert error < 1e-8
            self.roundtrip.append(float(error))

    def realize(self, target, previous=None):
        """Restrained minimization from the relaxed base and from the previous frame; keep the valid lower-energy one."""
        unit = self.unit
        for j, (ids, value) in enumerate(zip(TORSIONS, target)):
            self.restraint.setTorsionParameters(j, *ids, [float(value)])
        self.restraint.updateParametersInContext(self.context)
        attempts = []
        for initial in [self.base]+([previous] if previous is not None else []):
            x = initial.copy()
            for ids, value, subset in zip(TORSIONS, target, self.subsets):
                x = set_torsion(x, ids, value, subset)
            self.context.setPositions(x*unit.nanometer)
            self.context.applyConstraints(1e-8)
            self.minimizer.minimize(self.context, self.cfg['tolerance'], self.cfg['iterations'])
            state = self.context.getState(getPositions=True, getEnergy=True, groups={0})
            xyz = np.array(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
            energy = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
            achieved = np.array([dihedral(xyz, t) for t in TORSIONS])
            error = abs(periodic(achieved-target))*180/np.pi
            bond = np.array([np.linalg.norm(xyz[a]-xyz[b]) for a, b in self.bonds])
            same = bool(np.sign(chirality(xyz)) == self.reference_chirality)
            attempts.append({'positions': xyz, 'energy_kj_mol': energy, 'max_torsion_error_deg': float(error.max()),
                             'achieved_angles_rad': achieved.tolist(), 'omega_deg': [float(dihedral(xyz, t)*180/np.pi) for t in OMEGAS],
                             'same_chirality': same, 'max_relative_bond_change': float(np.max(abs(bond/self.base_lengths-1))),
                             'valid': bool(np.isfinite(energy) and error.max() < self.cfg['max_error_deg'] and same)})
        valid = [a for a in attempts if a['valid']]
        chosen = min(valid or attempts, key=lambda a: a['energy_kj_mol'])
        return chosen, [{k: v for k, v in a.items() if k != 'positions'} for a in attempts]

    def pairs(self):
        """Atom pairs more than three bonds apart."""
        graph = {i: set() for i in range(len(self.elements))}
        for i, j in self.bonds:
            graph[i].add(j)
            graph[j].add(i)
        out = []
        for i in graph:
            excluded, front = {i}, {i}
            for _ in range(3):
                front = set().union(*(graph[k] for k in front))-excluded
                excluded |= front
            out += [(i, j) for j in graph if j > i and j not in excluded]
        return out


def targets(path):
    """Saved knots and shortest-arc midpoints."""
    out = []
    for a, b in zip(path[:-1], path[1:]):
        out.extend([a, periodic(a+.5*periodic(b-a))])
    return np.array(out+[path[-1]])


def realize_path(realizer, path, pairs):
    """Frames for knots and midpoints, chained from the previous realization, with a path summary."""
    frames, xyz, previous = [], [], None
    for k, target in enumerate(targets(path)):
        row, attempts = realizer.realize(target, previous)
        previous = row.pop('positions')
        xyz.append(previous)
        frames.append({'time_ps': k*.05, 'is_original_knot': k % 2 == 0, 'target_angles_rad': np.asarray(target).tolist(),
                       **row, 'attempts': attempts})
    xyz = np.asarray(xyz)
    ratio = [np.linalg.norm(xyz[:, i]-xyz[:, j], axis=-1)/(BONDI[realizer.elements[i]]+BONDI[realizer.elements[j]]) for i, j in pairs]
    energy, omega = np.array([f['energy_kj_mol'] for f in frames]), np.array([f['omega_deg'] for f in frames])
    summary = {'configurations': len(frames), 'valid': sum(f['valid'] for f in frames),
               'chirality_preserved': sum(f['same_chirality'] for f in frames),
               'cis_peptide_counts': np.sum(abs(omega) < 90, axis=0).tolist(),
               'max_torsion_error_deg': max(f['max_torsion_error_deg'] for f in frames),
               'max_bond_change_percent': 100*max(f['max_relative_bond_change'] for f in frames),
               'max_energy_rise_kj_mol': float(energy.max()-energy[0]), 'initial_energy_kj_mol': float(energy[0]),
               'minimum_nonbonded_distance_over_vdw_sum': float(min(r.min() for r in ratio)),
               'severe_overlap_count': int(sum((r < realizer.cfg['overlap_ratio']).sum() for r in ratio)),
               'max_angular_step_deg': float(np.degrees(abs(periodic(np.diff(path, axis=0)))).max())}
    return frames, xyz, summary


def aggregate(rows):
    """Counts and worst cases over every realized configuration."""
    frames = [f for r in rows for f in r['frames']]
    omega = np.array([f['omega_deg'] for f in frames])
    return {'configurations': len(frames), 'valid': sum(f['valid'] for f in frames),
            'chirality_preserved': sum(f['same_chirality'] for f in frames),
            'max_torsion_error_deg': max(f['max_torsion_error_deg'] for f in frames),
            'max_bond_change_percent': 100*max(f['max_relative_bond_change'] for f in frames),
            'cis_configurations': int(np.any(abs(omega) < 90, axis=1).sum()),
            'severe_overlap_count': sum(r['summary']['severe_overlap_count'] for r in rows)}


def population(test, train, counts, runs, tasks, methods):
    """Per-path step, winding and occupancy diagnostics for every task, method and fit."""
    tau = test['masks_ab'].sum(-1).astype(int)
    steps = np.linalg.norm(periodic(train['paths_ab'][:, 1:]-train['paths_ab'][:, :-1]), axis=-1)[train['masks_ab'].astype(bool)]
    cutoff = float(np.quantile(steps, .99))
    rare = float(np.quantile(occupancy_cost(train['paths_ab'][:, :-1][train['masks_ab'].astype(bool)], counts), .99))
    rows = []
    for run, (meta, pred) in runs.items():
        for task in tasks:
            for method in methods:
                for j, i in enumerate(meta['eligibility'][task]['included_indices']):
                    p, target = pred[f'{task}__{method}'][j, :tau[i]+1], test['paths_ab'][i, :tau[i]+1]
                    step = np.linalg.norm(periodic(p[1:]-p[:-1]), axis=-1)
                    fe, turns = occupancy_cost(dense_path(p), counts), winding(p)
                    rows.append({'run': run, 'task': task, 'method': method, 'case_index': i, 'steps': int(tau[i]),
                                 'max_abs_phi_deg': float(abs(p[:, 0]).max()*180/np.pi), 'max_step_deg': float(step.max()*180/np.pi),
                                 'large_step_fraction': float((step > cutoff).mean()), 'train_step_99pct_deg': cutoff*180/np.pi,
                                 'dense_fe_max': float(fe.max()), 'dense_fe_mean': float(fe.mean()),
                                 'rare_region_fraction': float((fe > rare).mean()), 'train_fe_99pct': rare,
                                 'phi_winding': int(turns[0]), 'psi_winding': int(turns[1]),
                                 'full_path_mse': float((periodic(target-p)**2).mean())})
    return rows


def select_examples(test, rows, meta, run, min_steps):
    """Median-duration bounded completion; largest mean unconstrained phi excursion; largest mean bounded occupancy."""
    tau = test['masks_ab'].sum(-1).astype(int)
    eligible = np.array(meta['eligibility']['completion_fence']['included_indices'])
    order = np.lexsort((eligible, tau[eligible]))
    long = np.where(tau >= min_steps)[0]
    mean = lambda task, key: np.array([np.mean([r[key] for r in rows if r['task'] == task and r['method'] == 'leap' and r['case_index'] == int(i)])
                                       for i in long])
    return [{'role': 'representative_completion', 'task': 'completion_fence', 'case_index': int(eligible[order[len(order)//2]]), 'run': run},
            {'role': 'constraint_change', 'task': 'endpoints', 'case_index': int(long[np.argmax(mean('endpoints', 'max_abs_phi_deg'))]), 'run': run},
            {'role': 'high_energy_stress', 'task': 'endpoints_fence', 'case_index': int(long[np.argmax(mean('endpoints_fence', 'dense_fe_max'))]), 'run': run}]
