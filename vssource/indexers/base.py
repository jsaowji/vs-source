from __future__ import annotations

import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from hashlib import md5
from typing import Any, Callable, ClassVar, Iterable, Literal, Protocol, Sequence

from vstools import (
    MISSING, ChromaLocationT, ColorRangeT, CustomRuntimeError, DataType, FieldBasedT, MatrixT, MissingT, PrimariesT,
    SPath, SPathLike, TransferT, core, initialize_clip, inject_self, to_arr, vs
)

from ..dataclasses import IndexFileType

__all__ = [
    'Indexer', 'ExternalIndexer',

    'VSSourceFunc'
]


class VSSourceFunc(Protocol):
    def __call__(self, path: DataType, *args: Any, **kwargs: Any) -> vs.VideoNode:
        ...


class Indexer(ABC):
    """Abstract indexer interface."""

    index_folder_name = '.vssource'

    _source_func: ClassVar[Callable[..., vs.VideoNode]]

    def __init__(self, *, force: bool = True, **kwargs: Any) -> None:
        super().__init__()

        self.force = force
        self.indexer_kwargs = kwargs

    @classmethod
    def _split_lines(cls, buff: list[str]) -> tuple[list[str], list[str]]:
        return buff[:(split_idx := buff.index(''))], buff[split_idx + 1:]

    @classmethod
    def get_joined_names(cls, files: list[SPath]) -> str:
        return '_'.join([file.name for file in files])

    @classmethod
    def get_videos_hash(cls, files: list[SPath]) -> str:
        lenght = sum(file.stat().st_size for file in files)
        to_hash = lenght.to_bytes(32, 'little') + cls.get_joined_names(files).encode()
        return md5(to_hash).hexdigest()

    @property
    def source_func(self) -> VSSourceFunc:
        return self._source_func

    def _source(
        self, clips: Iterable[vs.VideoNode],
        bits: int | None = None,
        matrix: MatrixT | None = None,
        transfer: TransferT | None = None,
        primaries: PrimariesT | None = None,
        chroma_location: ChromaLocationT | None = None,
        color_range: ColorRangeT | None = None,
        field_based: FieldBasedT | None = None
    ) -> vs.VideoNode:
        clips = list(clips)

        if len(clips) == 1:
            clip = clips[0]
        else:
            clip = core.std.Splice(clips)

        return initialize_clip(
            clip, bits, matrix, transfer, primaries, chroma_location, color_range, field_based
        )

    @inject_self
    def source(
        self, file: SPathLike | Sequence[SPathLike],
        bits: int | None = None, *,
        matrix: MatrixT | None = None,
        transfer: TransferT | None = None,
        primaries: PrimariesT | None = None,
        chroma_location: ChromaLocationT | None = None,
        color_range: ColorRangeT | None = None,
        field_based: FieldBasedT | None = None,
        **kwargs: Any
    ) -> vs.VideoNode:
        return self._source(
            [self.source_func(SPath(f).to_str(), **kwargs) for f in to_arr(file)],  # type: ignore
            bits, matrix, transfer, primaries, chroma_location, color_range, field_based
        )


class ExternalIndexer(Indexer):
    _bin_path: ClassVar[str]
    _ext: ClassVar[str]

    _default_args: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self, *, bin_path: SPathLike | MissingT = MISSING, ext: str | MissingT = MISSING,
        force: bool = True, default_out_folder: SPathLike | Literal[False] | None = None,
        **kwargs: Any
    ) -> None:
        super().__init__(force=force, **kwargs)

        if bin_path is MISSING:
            bin_path = self._bin_path

        if ext is MISSING:
            ext = self._ext

        self.bin_path = SPath(bin_path)
        self.ext = ext
        self.default_out_folder = default_out_folder

    @abstractmethod
    def get_cmd(self, files: list[SPath], output: SPath) -> list[str]:
        """Returns the indexer command"""
        raise NotImplementedError

    @abstractmethod
    def get_info(self, index_path: SPath, file_idx: int = 0) -> IndexFileType:
        """Returns info about the indexing file"""
        raise NotImplementedError

    @abstractmethod
    def update_video_filenames(self, index_path: SPath, filepaths: list[SPath]) -> None:
        raise NotImplementedError

    def _get_bin_path(self) -> SPath:
        if not (bin_path := shutil.which(str(self.bin_path))):
            raise FileNotFoundError(f'Indexer: `{self.bin_path}` was not found!')
        return SPath(bin_path)

    def _run_index(self, files: list[SPath], output: SPath, cmd_args: Sequence[str]) -> None:
        output.mkdirp()

        proc = subprocess.Popen(
            list(map(str, self.get_cmd(files, output) + list(cmd_args) + list(self._default_args))),
            text=True, encoding='utf-8', shell=True, cwd=output.get_folder().to_str()
        )

        status = proc.wait()

        if status:
            stderr = stdout = ''

            if proc.stderr:
                stderr = proc.stderr.read().strip()
                if stderr:
                    stderr = f'\n\t{stderr}'

            if proc.stdout:
                stdout = proc.stdout.read().strip()
                if stdout:
                    stdout = f'\n\t{stdout}'

            raise CustomRuntimeError(
                f"There was an error while running the {self.bin_path} command!: {stderr}{stdout}"
            )

    def get_out_folder(
        self, output_folder: SPathLike | Literal[False] | None = None, file: SPath | None = None
    ) -> SPath:
        if output_folder is None:
            return SPath(file).get_folder() if file else self.get_out_folder(False)

        if not output_folder:
            return SPath(tempfile.gettempdir())

        return SPath(output_folder)

    def get_idx_file_path(self, path: SPath) -> SPath:
        return path.with_suffix(f'.{self.ext}')

    def file_corrupted(self, index_path: SPath) -> None:
        if self.force:
            try:
                index_path.unlink()
            except OSError:
                raise CustomRuntimeError("Index file corrupted, tried to delete it and failed.", self.__class__)
        else:
            raise CustomRuntimeError("Index file corrupted! Delete it and retry.", self.__class__)

    def index(
        self, files: Sequence[SPath], force: bool = False, split_files: bool = False,
        output_folder: SPathLike | Literal[False] | None = None, *cmd_args: str
    ) -> list[SPath]:
        files = to_arr(files)

        if len(unique_folders := list(set([f.get_folder().to_str() for f in files]))) > 1:
            return [
                c for s in (
                    self.index(
                        [f for f in files if f.get_folder().to_str() == folder],
                        force, split_files, output_folder
                    )
                    for folder in unique_folders
                ) for c in s
            ]

        dest_folder = self.get_out_folder(output_folder, files[0])

        files = list(sorted(set(files)))

        hash_str = self.get_videos_hash(files)

        def _index(files: list[SPath], output: SPath) -> None:
            if output.is_file():
                if output.stat().st_size == 0 or force:
                    output.unlink()
                else:
                    return self.update_video_filenames(output, files)

            return self._run_index(files, output, cmd_args)

        if not split_files:
            output = self.get_video_idx_path(dest_folder, hash_str, 'JOINED' if len(files) > 1 else 'SINGLE')
            _index(files, output)
            return [output]

        outputs = [self.get_video_idx_path(dest_folder, hash_str, file.name) for file in files]

        for file, output in zip(files, outputs):
            _index([file], output)

        return outputs

    def get_video_idx_path(self, folder: SPath, file_hash: str, video_name: SPathLike) -> SPath:
        vid_name = SPath(video_name).stem
        filename = '_'.join([file_hash, vid_name])
        return self.get_idx_file_path(folder / self.index_folder_name / filename)

    @inject_self
    def source(  # type: ignore
        self, file: SPathLike | Sequence[SPathLike],
        bits: int | None = None, *,
        matrix: MatrixT | None = None,
        transfer: TransferT | None = None,
        primaries: PrimariesT | None = None,
        chroma_location: ChromaLocationT | None = None,
        color_range: ColorRangeT | None = None,
        field_based: FieldBasedT | None = None,
        **kwargs: Any
    ) -> vs.VideoNode:
        index_files = self.index([SPath(f) for f in to_arr(file)])  # type: ignore

        return self._source(
            (self.source_func(idx_filename.to_str(), **kwargs) for idx_filename in index_files),
            bits, matrix, transfer, primaries, chroma_location, color_range, field_based
        )
