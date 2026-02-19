from typing import (
    Any, Callable, Dict, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Type, Union
)
from pydantic import BaseModel, Field

class ListExperimentsParams(BaseModel):
    """No parameters for listing experiments."""
    pass

class ListRunsParams(BaseModel):
    experiment_ids: List[str] = Field(..., description="IDs of the experiments to list runs for.")
    max_results: int = Field(20, description="Maximum number of runs to return.")

class GetRunMetricsParams(BaseModel):
    run_id: str = Field(..., description="ID of the run to get metrics for.")

class GetRunParamsParams(BaseModel):
    run_id: str = Field(..., description="ID of the run to get parameters for.")

class FindBestRunByMetricParams(BaseModel):
    experiment_ids: List[str] = Field(..., description="IDs of the experiments to search in.")
    metric: str = Field(..., description="Metric name to optimize.")
    mode: Literal['max', 'min'] = Field('max', description="Whether to maximize or minimize the metric.")

class CheckExperimentGeneralizationParams(BaseModel):
    experiment_name: str = Field(..., description="Name of the experiment to check.")
    metric: str = Field('loss', description="Metric to compare between train and test.")
    threshold: Optional[float] = Field(None, description="Threshold for difference between test and train metrics.")