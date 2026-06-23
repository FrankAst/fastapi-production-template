from fastapi import Request, status
from fastapi.responses import JSONResponse

from app.utils import CsvContentError, CsvFormatError, CsvSizeError


def csv_format_handler(_: Request, exc: CsvFormatError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )


def csv_size_handler(_: Request, exc: CsvSizeError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        content={"detail": str(exc)},
    )


def csv_content_handler(_: Request, exc: CsvContentError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)},
    )
