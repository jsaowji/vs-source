from __future__ import annotations

from dataclasses import dataclass
import warnings
from copy import deepcopy
from itertools import count
from typing import Sequence

from vstools import CustomRuntimeError, T, flatten, remap_frames, vs

__all__ = [
    'apply_rff_array', 'apply_rff_video',
    'cut_array_on_ranges'
]

@dataclass
class Field:
    n: int
    is_tf: bool
    is_repeat: bool
    prog: bool
    user_data: object

def rff_frames_to_fields(rff: list[int], tff: list[int], prog: list[int], prog_seq: list[int], user_data: list[T]):
    fields = list[Field]()

    for i, current_prg_seq, current_prg, current_rff, current_tff,current_ud in zip(count(), prog_seq, prog, rff, tff, user_data):
        if not current_prg_seq:
            first_field  = [2*i+1, 2*i+0][current_tff]
            second_field = [2*i+0, 2*i+1][current_tff]

            fields += [
                Field(first_field, current_tff, False,False,current_ud),
                Field(second_field, not current_tff, False,False,current_ud)
            ]

            if current_rff:
                assert current_prg
                repeat_field = deepcopy(fields[-2])
                repeat_field.is_repeat = True
                fields.append(repeat_field)
        else:
            assert current_prg

            field_count = 1
            if current_rff:
                field_count += 1 + int(current_tff)

            #maybe set is_repeat even for progressive repeats ?
            fields += [
                Field(2 * i,     True, False,True,current_ud),
                Field(2 * i + 1, False, False,True,current_ud),
            ] * field_count

    #There might be a need to make this adjustable
    fixmode_invalid_tff_parity: int = 1


    a = 0
    while a < (len(fields) // 2) * 2:
        tf = fields[a]
        bf = fields[a+1]
        if tf.is_tf == bf.is_tf:
            warnings.warn(f'Invalid field transition at {a / 2} {tf} {bf}')

            if fixmode_invalid_tff_parity == 0:
                bf.is_tf = not bf.is_tf
            else:
                fc = deepcopy(tf)
                fc.is_tf = not fc.is_tf
                fields.insert(a+1,fc)
        a += 2

    if (len(fields) % 2) != 0:
        warnings.warn('uneven amount of fields removing last\n')
        fields = fields[:-1]

    return fields
    


def apply_rff_array(old_array: list[T], rff: list[int], tff: list[int], prog: list[int], prog_seq: list[int]) -> list[T]:
    return list([f.user_data for f in rff_frames_to_fields(rff,tff,prog,prog_seq,old_array)][1::2])

def apply_rff_video(
    node: vs.VideoNode, rff: list[int], tff: list[int], prog: list[int], prog_seq: list[int]
) -> vs.VideoNode:
    assert len(node) == len(rff) == len(tff) == len(prog) == len(prog_seq)

    tfffs = node.std.RemoveFrameProps(['_FieldBased', '_Field']).std.SeparateFields(True)
    fields = rff_frames_to_fields(rff,tff,prog,prog_seq,list(range(len(tff))))

    for fcurr, fnext in zip(fields[::2], fields[1::2]):
        if fcurr.is_tf == fnext.is_tf:
            raise CustomRuntimeError(
                f'Found invalid stream with two consecutive {"top" if fcurr.is_tf else "bottom"} fields!'
            )

    final = remap_frames(tfffs, [x.n for x in fields])

    def _set_field(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        f = f.copy()

        f.props.pop('_FieldBased', None)
        f.props._Field = fields[n].is_tf

        return f

    final = final.std.ModifyFrame(final, _set_field)

    woven = final.std.DoubleWeave()[::2]

    def _set_repeat(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        f = f.copy()
        if fields[n * 2].is_repeat:
            f.props['RepeatedField'] = 1
        elif fields[n * 2 + 1].is_repeat:
            f.props['RepeatedField'] = 0
        else:
            f.props['RepeatedField'] = -1
        return f

    woven = woven.std.ModifyFrame(woven, _set_repeat)

    # TODO: this seems to not work or atleast useless since its disable for non progressive sequence which is rare
    def _update_progressive(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        fout = f.copy()

        tf = fields[n * 2]
        bf = fields[n * 2 + 1]

        if tf.prog and bf.prog:
            fout.props['_FieldBased'] = 0

        return fout

    return woven.std.ModifyFrame(woven, _update_progressive)


def cut_array_on_ranges(array: list[T], ranges: list[tuple[int, int]]) -> list[T]:
    return [array[i] for i in flatten([range(rrange[0], rrange[1] + 1) for rrange in ranges])]  # type: ignore
