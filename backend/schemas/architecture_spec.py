from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

class LayerSpec(BaseModel):
    type: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    channels: Optional[int] = Field(None, gt=0)
    kernel_size: Optional[int] = Field(None, gt=0)
    stride: Optional[int] = Field(None, gt=0)
    heads: Optional[int] = Field(None, gt=0)
    hidden_size: Optional[int] = Field(None, gt=0)

class ArchitectureSpec(BaseModel):
    family: str = Field(..., min_length=1)
    input_shape: List[int] = Field(..., min_items=1)
    layers: List[LayerSpec] = Field(..., min_items=1, max_items=200)
    
    @field_validator('family')
    @classmethod
    def family_must_be_known(cls, v):
        known = {'transformer', 'cnn', 'vit', 'unet', 'resnet', 'unknown', 'generic'}
        return v.lower() if v.lower() in known else 'unknown'
