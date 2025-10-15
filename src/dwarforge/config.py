from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

# ---------- schema models ----------


class LoggingCfg(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str
    level: str


class MonitoringMemory(BaseModel):
    model_config = ConfigDict(extra='forbid')
    enable: bool
    interval_s: int
    aggregation_s: int


class MonitoringProgress(BaseModel):
    model_config = ConfigDict(extra='forbid')
    database_name: str
    resume: bool


class Monitoring(BaseModel):
    model_config = ConfigDict(extra='forbid')
    memory: MonitoringMemory
    progress: MonitoringProgress


class Runtime(BaseModel):
    model_config = ConfigDict(extra='forbid')
    num_cores: int
    prefetch_factor: int
    anchor_band: str
    considered_bands: List[str]
    use_full_resolution: bool
    process_all_available: bool
    process_only_known_dwarfs: bool


class CombinationCfg(BaseModel):
    model_config = ConfigDict(extra='forbid')
    bands_to_combine: List[str]
    max_match_sep_arcsec: float
    negatives_per_positive: int
    accumulate_lsb_to_h5: bool
    aggregate_cutouts: bool
    aggregate_objects_per_file: int
    combine_cutouts: bool
    process_groups_only: bool
    group_tiles_csv: str
    lsb_cutout_path: Path


class H5AggregationCfg(BaseModel):
    model_config = ConfigDict(extra='forbid')
    objects_per_file: int
    label_filters: list[str | int] | None
    in_file_suffix: str
    out_file_prefix: str
    number_of_tiles: int | None
    preprocess_cutouts: bool
    preprocessing_mode: Literal['vis', 'training']
    tile_df_file: Path


class Tiles(BaseModel):
    model_config = ConfigDict(extra='forbid')
    update_tiles: bool
    build_new_kdtree: bool
    show_tile_statistics: bool
    band_constraint: int
    print_per_tile_availability: bool = Field(default=False)


class Detection(BaseModel):
    model_config = ConfigDict(extra='forbid')
    mu_limit: float
    re_limit: float
    mto: MTOCfg


class MTOCfg(BaseModel):
    model_config = ConfigDict(extra='forbid')
    script_path: Path
    with_segmap: bool
    move_factor: float
    min_distance: float


class Cutouts(BaseModel):
    model_config = ConfigDict(extra='forbid')
    create: bool
    plot_random: bool
    size_px: int
    segmentation_mode: str


class InputsDataFrame(BaseModel):
    model_config = ConfigDict(extra='forbid')
    path: Path
    columns: ColumnMap


class Inputs(BaseModel):
    model_config = ConfigDict(extra='forbid')
    source: Literal['all_available', 'tiles', 'coordinates', 'dataframe']
    tiles: list[tuple[int, int]] = Field(default_factory=list)
    coordinates: list[tuple[float, float]] = Field(default_factory=list)
    dataframe: InputsDataFrame


class ColumnMap(BaseModel):
    model_config = ConfigDict(extra='forbid')
    ra: str
    dec: str
    id: str


class DwarfCatalogCfg(BaseModel):
    model_config = ConfigDict(extra='forbid')
    path: Path
    columns: ColumnMap


class CatalogCfg(BaseModel):
    model_config = ConfigDict(extra='forbid')
    dwarf: DwarfCatalogCfg


class PathsByMachineEntry(BaseModel):
    model_config = ConfigDict(extra='forbid')
    root_dir_main: Path
    root_dir_data: Path
    redshift_class_catalog: Path
    download_directory: Path
    cutout_directory: Path


class PathsCommon(BaseModel):
    model_config = ConfigDict(extra='forbid')
    table_dirname: str
    tile_info_dirname: str
    figure_dirname: str
    logs_dirname: str
    database_dirname: str
    aggregate_dirname: str


class Band(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str
    band: str
    vos: str
    suffix: str
    delimiter: str
    fits_ext: int
    zfill: int
    zp: float


class RawConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    machine: str
    logging: LoggingCfg
    monitoring: Monitoring
    runtime: Runtime
    combination: CombinationCfg
    tiles: Tiles
    detection: Detection
    cutouts: Cutouts
    inputs: Inputs
    paths_common: PathsCommon
    paths_by_machine: Dict[str, PathsByMachineEntry]
    bands: Dict[str, Band]
    catalog: CatalogCfg
    h5_aggregation: H5AggregationCfg


class PathsResolved(BaseModel):
    model_config = ConfigDict(extra='forbid')
    root_dir_main: Path
    root_dir_data: Path
    download_directory: Path
    cutout_directory: Path
    table_directory: Path
    tile_info_directory: Path
    figure_directory: Path
    log_directory: Path
    database_directory: Path
    progress_db_path: Path
    redshift_class_catalog: Path
    aggregate_directory: Path


class Settings(BaseModel):
    model_config = ConfigDict(extra='forbid')
    machine: str
    logging: LoggingCfg
    monitoring: Monitoring
    runtime: Runtime
    combination: CombinationCfg
    tiles: Tiles
    detection: Detection
    cutouts: Cutouts
    inputs: Inputs
    bands: Dict[str, Band]
    paths: PathsResolved
    catalog: CatalogCfg
    h5_aggregation: H5AggregationCfg

    @model_validator(mode='after')
    def _validate(self) -> 'Settings':
        if self.runtime.anchor_band not in self.runtime.considered_bands:
            raise ValueError('runtime.anchor_band must appear in runtime.considered_bands')
        for b in self.runtime.considered_bands:
            if b not in self.bands:
                raise ValueError(f'Unknown band in considered_bands: {b}')
        missing = set(self.combination.bands_to_combine) - set(self.runtime.considered_bands)
        if missing:
            raise ValueError(
                f'combination.bands_to_combine not in runtime.considered_bands: {sorted(missing)}'
            )
        if self.combination.lsb_cutout_path.suffix.lower() != '.h5':
            raise ValueError('combination.lsb_cutout_path must end with .h5')
        if self.combination.max_match_sep_arcsec <= 0:
            raise ValueError('combination.max_match_sep_arcsec must be > 0')
        if not self.combination.lsb_cutout_path.is_absolute():
            self.combination.lsb_cutout_path = (
                self.paths.aggregate_directory / self.combination.lsb_cutout_path
            )
        if not self.h5_aggregation.tile_df_file.is_absolute():
            self.h5_aggregation.tile_df_file = (
                self.paths.table_directory / self.h5_aggregation.tile_df_file
            )
        if self.cutouts.segmentation_mode not in {'concatenate', 'mask', 'none'}:
            raise ValueError('cutouts.segmentation_mode must be one of: concatenate | mask | none')
        if self.runtime.num_cores <= 1:
            self.runtime.num_cores = 1
        if self.combination.group_tiles_csv and not Path(self.combination.group_tiles_csv).exists():
            raise ValueError(
                f'combination.group_tiles_csv not found: {self.combination.group_tiles_csv}'
            )
        if self.h5_aggregation.preprocessing_mode not in {'vis', 'training'}:
            raise ValueError("h5_aggregation.preprocessing_mode must be 'vis' or 'training'")
        return self


# ---------- loader ----------


def load_settings(
    path: str = 'configs/default.yaml', machine_override: Optional[str] = None
) -> Settings:
    with open(path, 'r', encoding='utf-8') as f:
        raw = RawConfig.model_validate(yaml.safe_load(f))
    machine = machine_override or raw.machine
    if machine not in raw.paths_by_machine:
        raise ValueError(f"Unknown machine '{machine}'. Options: {list(raw.paths_by_machine)}")
    pm = raw.paths_by_machine[machine]
    pc = raw.paths_common

    root = pm.root_dir_main
    table = root / pc.table_dirname
    tile_info = root / pc.tile_info_dirname
    figure = root / pc.figure_dirname
    logs = root / pc.logs_dirname
    dbdir = root / pc.database_dirname
    progress_db = dbdir / raw.monitoring.progress.database_name
    aggregate = root / pc.aggregate_dirname

    paths = PathsResolved(
        root_dir_main=root,
        root_dir_data=pm.root_dir_data,
        download_directory=pm.download_directory,
        cutout_directory=pm.cutout_directory,
        table_directory=table,
        tile_info_directory=tile_info,
        figure_directory=figure,
        log_directory=logs,
        database_directory=dbdir,
        progress_db_path=progress_db,
        redshift_class_catalog=pm.redshift_class_catalog,
        aggregate_directory=aggregate,
    )

    cat = CatalogCfg(
        dwarf=DwarfCatalogCfg(
            path=(table / raw.catalog.dwarf.path), columns=raw.catalog.dwarf.columns
        )
    )

    detection = Detection(
        mu_limit=raw.detection.mu_limit,
        re_limit=raw.detection.re_limit,
        mto=MTOCfg(
            script_path=root / raw.detection.mto.script_path,
            with_segmap=raw.detection.mto.with_segmap,
            move_factor=raw.detection.mto.move_factor,
            min_distance=raw.detection.mto.min_distance,
        ),
    )

    return Settings(
        machine=machine,
        logging=raw.logging,
        monitoring=raw.monitoring,
        runtime=raw.runtime,
        combination=raw.combination,
        tiles=raw.tiles,
        detection=detection,
        cutouts=raw.cutouts,
        inputs=raw.inputs,
        bands=raw.bands,
        paths=paths,
        catalog=cat,
        h5_aggregation=raw.h5_aggregation,
    )


def settings_to_jsonable(cfg: Settings) -> Dict[str, Any]:
    # Pydantic v2: turn Paths into strings etc.
    return cast(Dict[str, Any], cfg.model_dump(mode='json'))


def ensure_runtime_dirs(cfg: Settings) -> None:
    """
    Ensure that all runtime directories exist.
    """
    for p in (
        cfg.paths.table_directory,
        cfg.paths.tile_info_directory,
        cfg.paths.figure_directory,
        cfg.paths.log_directory,
        cfg.paths.database_directory,
        cfg.paths.cutout_directory,
        cfg.paths.aggregate_directory,
    ):
        p.mkdir(parents=True, exist_ok=True)
