from pydantic import BaseModel, RootModel, model_validator, model_serializer
from typing import Optional
from enum import StrEnum


class Category(StrEnum):
    Caption = "Caption"
    Footnote = "Footnote"
    Formula = "Formula"
    ListItem = "List-item"
    PageFooter = "Page-footer"
    PageHeader = "Page-header"
    Picture = "Picture"
    SectionHeader = "Section-header"
    Table = "Table"
    Text = "Text"
    Title = "Title"


class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

    @model_validator(mode="before")
    @classmethod
    def parse_from_list(cls, v):
        if isinstance(v, (list, tuple)) and len(v) == 4:
            x1, y1, x2, y2 = v
            return {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
        return v

    @model_serializer
    def to_list(self):
        return [self.x1, self.y1, self.x2, self.y2]


class Element(BaseModel):
    bbox: BBox
    category: Category
    text: Optional[str] = None


class Result(RootModel):
    root: list[Element]
