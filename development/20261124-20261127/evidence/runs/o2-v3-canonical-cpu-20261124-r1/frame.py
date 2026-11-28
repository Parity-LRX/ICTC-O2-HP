"""Derive the physical-reflection transport from the actual frozen frame constructor."""
import os

import torch
from common import start, finish, save, difference, parameter_hashes


def main():
    output, net, checkpoint, prior = start('frame')
    role = prior.load_parameter_role(net, checkpoint, 'checkpoint_original')
    before = parameter_hashes(net.model)
    from e3nn import o3
    from experimental.models.equiformer_v3.so3 import CoefficientMappingModule, SO3Rotation
    from experimental.models.equiformer_v3.edge_rot_mat import init_edge_rot_mat as original_frame
    from rotation_compat import init_edge_rot_mat as actual_frame
    from experimental.models.equiformer_v3.edge_rot_mat import _ROTATION_MASK_THRESHOLD

    mapping = CoefficientMappingModule(4, 2)
    full = CoefficientMappingModule(4, 4)
    indices = full.coefficient_idx(4, 2)
    permutation = torch.zeros(19, 25)
    permutation[:, indices] = mapping.to_m
    tangent = torch.tensor([[0., 1.], [0., 0.], [1., 0.]])  # (local z, local x); y is the edge axis.
    axis = torch.tensor([0., 1., 0.])
    regular = torch.tensor([[.7, -.4, 1.2], [-1.1, .8, .3], [1.4, .9, -.6],
                            [-.7, .4, -1.2], [1.1, -.8, -.3], [-1.4, -.9, .6],
                            [.7, -.4, 1.2], [4.9, -.4, 1.2]])
    mask = torch.tensor([[0., 1., 0.], [0., -1., 0.], [.0005, 1., .0003],
                         [-.0005, -1., -.0003], [.002, 1., .001], [-.002, -1., -.001]])
    vectors = torch.cat((regular, mask))
    n = len(vectors)
    length = vectors.square().sum(-1).sqrt()
    unit = vectors/length[:, None]
    branch = unit[:, 1].abs() > _ROTATION_MASK_THRESHOLD
    sphere_rotation = SO3Rotation(4, 2, use_rotation_mask=True)

    def frames(v, seed=20261124, compat=True):
        torch.manual_seed(seed)
        fn = actual_frame if compat else original_frame
        return fn(v.clone(), use_rotation_mask=True)

    def wigner(r):
        sphere_rotation.set_wigner(r)
        return sphere_rotation.wigner.clone(), sphere_rotation.wigner_inv.clone()

    def representation(matrix):
        values = [o3.Irrep(l, (-1)**l).D_from_matrix(matrix) for l in range(5)]
        out = matrix.new_zeros((matrix.shape[0], 25, 25))
        for l, value in enumerate(values):
            out[:, l*l:(l+1)**2, l*l:(l+1)**2] = value
        return out

    with torch.no_grad():
        base = frames(vectors)
        original = frames(vectors, compat=False)
        original_comparison = difference(base, original)
        wb, ib = wigner(base)
        direct = permutation@representation(base)
        native_vs_representation = difference(wb, direct)
        e_identity = torch.eye(3).expand(n, -1, -1)
        base_audit = {'orthogonality': difference(base@base.transpose(1, 2), e_identity),
                      'determinant': torch.linalg.det(base).tolist(),
                      'maps_actual_direction_to_axis': difference((base@unit[:, :, None]).squeeze(2), axis.expand(n, -1)),
                      'compat_vs_original_forward': original_comparison,
                      'native_Wigner_vs_e3nn_representation': native_vs_representation,
                      'mask_branch': branch.tolist()}
        normal = torch.tensor([1., 2., -1.]); normal = normal/normal.norm()
        mirrors = {'mirror_x': torch.diag(torch.tensor([-1., 1., 1.])),
                   'mirror_z': torch.diag(torch.tensor([1., 1., -1.])),
                   'householder_1_2_m1': torch.eye(3)-2*normal[:, None]*normal[None, :]}
        records, tensors = {}, {'vectors': vectors, 'base_frames': base, 'base_W': wb, 'base_inv': ib}
        for name, mirror in mirrors.items():
            changed_vectors = vectors@mirror.T
            changed = frames(changed_vectors)
            wc, ic = wigner(changed)
            h = changed@mirror@base.transpose(1, 2)
            q = tangent.T@h@tangent
            dh = representation(h)
            local = permutation@dh@permutation.T
            dm = representation(mirror[None].expand(n, -1, -1))
            lhs, rhs = wc@dm, local@wb
            left_inverse, right_inverse = ic@local, dm@ib
            changed_unit = changed_vectors/changed_vectors.norm(dim=-1, keepdim=True)
            changed_branch = changed_unit[:, 1].abs() > _ROTATION_MASK_THRESHOLD
            masks_same = branch == changed_branch
            regular_rows = (~branch)&(~changed_branch)
            selected = torch.nonzero(regular_rows).flatten()
            l_to_m_leak = local.clone()
            for frequency in range(3):
                offset = 0 if frequency == 0 else (5 if frequency == 1 else 13)
                width = 5 if frequency == 0 else 2*(5-frequency)
                l_to_m_leak[:, offset:offset+width, offset:offset+width] = 0
            actual = {'H_determinant': torch.linalg.det(h).tolist(),
                      'Q_determinant': torch.linalg.det(q).tolist(),
                      'H_axis_max_abs': (h@axis-axis).abs().amax(1).tolist(),
                      'Q_orthogonality': difference(q@q.transpose(1, 2), torch.eye(2).expand(n, -1, -1)),
                      'all_rows_forward_transport': difference(lhs, rhs),
                      'all_rows_inverse_transport': difference(left_inverse, right_inverse),
                      'regular_row_indices': selected.tolist(),
                      'regular_forward_transport': difference(lhs[selected], rhs[selected]),
                      'regular_inverse_transport': difference(left_inverse[selected], right_inverse[selected]),
                      'same_mask_branch': masks_same.tolist(), 'reflected_mask_branch': changed_branch.tolist(),
                      'off_frequency_action_max_abs': float(l_to_m_leak[regular_rows].abs().max()),
                      'phase_angle_rad': torch.atan2(q[:, 1, 0], q[:, 0, 0]).tolist()}
            # Compare all l copies at fixed m against the canonical planar action.
            block_records = []
            from common import load, WS
            o2math = load('frame_o2_math_'+name, WS/'baselines/icml-o2-stage1-20261124-r1/payload/operator/upstream/official/src/ictc_o2_hp/o2_math.py')
            for m, offset, count in ((1, 5, 4), (2, 13, 3)):
                canonical = o2math.o2_representation(q, ('freq', m))
                maximum = 0.
                for copy in range(count):
                    ids = torch.tensor([offset+copy, offset+count+copy])
                    observed = local[:, ids][:, :, ids]
                    maximum = max(maximum, float((observed[regular_rows]-canonical[regular_rows]).abs().max()))
                block_records.append({'m': m, 'copies': count, 'canonical_planar_action_max_abs': maximum})
            actual['same_m_l_copy_alignment'] = block_records
            second = frames(changed_vectors@mirror.T)
            h_second = second@mirror@changed.transpose(1, 2)
            actual['double_reflection_frame_cocycle'] = difference(h_second@h, e_identity)
            records[name] = actual
            tensors[name] = {'reflected_frames': changed, 'H': h, 'Q': q, 'local_action': local,
                             'W_reflected': wc, 'inverse_reflected': ic,
                             'physical_mirror': mirror, 'global_action': dm}

        other_gauge = frames(vectors, seed=20261125)
        gauge = other_gauge@base.transpose(1, 2)
        reverse = frames(-vectors)
        reverse_transition = reverse@base.transpose(1, 2)
        reverse_tangent = tangent.T@reverse_transition@tangent
        distinctions = {
            'same_edge_other_seed': {'determinant': torch.linalg.det(gauge).tolist(),
                'axis_error_max': float((gauge@axis-axis).abs().max())},
            'directed_edge_reversal': {'determinant_3d': torch.linalg.det(reverse_transition).tolist(),
                'maps_axis_to_negative_max_abs': float((reverse_transition@axis+axis).abs().max()),
                'tangent_determinant': torch.linalg.det(reverse_tangent).tolist()},
            'rule': 'Gauge change fixes +y and is proper; physical mirror fixes +y and is improper; directed-edge reversal flips y and is proper. Preserve both directed contributions and original endpoint/radial semantics.'}
        tensors['other_seed_frames'] = other_gauge
        tensors['reverse_frames'] = reverse
        torch.save(tensors, output/'FRAME-TENSORS.pt')
    assert before == parameter_hashes(net.model)
    result = {'status': 'ACTUAL_FRAME_TRANSPORT_DERIVED_CPU_DIAGNOSTICS', 'job_id': os.environ['SLURM_JOB_ID'],
        'parameter_role_identity': role, 'all_292_parameters_unchanged': len(before) == 292,
        'base': base_audit, 'mirrors': records, 'distinctions': distinctions,
        'derived_action': 'H_e=R(Mv) M R(v)^T; Q_e=T^T H_e T, T=(e_z,e_x); at m>0 z_m -> exp(i m phi_e) C_m conj(z_m), C_m conj(C_m)=I',
        'scope': 'Actual native forward frames, Wigner ordering, inverse rescale, masks and random gauges; covariance diagnostic only, no derivative/backend qualification',
        'gpu_jobs': 0, 'gpu_hours': 0}
    finish('frame', output, result)


if __name__ == '__main__':
    main()
