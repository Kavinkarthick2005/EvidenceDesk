"""Domain exception handlers for FastAPI application."""

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.generation.exceptions import GenerationError
from app.services.exceptions import (
    DocumentNotFoundError,
    DocumentPersistenceError,
    DocumentServiceError,
    DocumentSizeLimitError,
    DocumentStorageError,
    DocumentValidationError,
    EmptyDocumentError,
    InvalidQueryError,
    RetrievalError,
    UnsupportedDocumentTypeError,
)

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Register application domain exception handlers with the FastAPI application.

    Translates domain service exceptions into predictable, standard HTTP responses
    without leaking internal database details, SQL queries, or filesystem paths.
    """

    @app.exception_handler(DocumentNotFoundError)
    async def document_not_found_handler(
        request: Request,
        exc: DocumentNotFoundError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": str(exc)},
        )

    @app.exception_handler(DocumentSizeLimitError)
    async def document_size_limit_handler(
        request: Request,
        exc: DocumentSizeLimitError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            content={"detail": str(exc)},
        )

    @app.exception_handler(UnsupportedDocumentTypeError)
    async def unsupported_document_type_handler(
        request: Request,
        exc: UnsupportedDocumentTypeError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            content={"detail": str(exc)},
        )

    @app.exception_handler(EmptyDocumentError)
    async def empty_document_handler(
        request: Request,
        exc: EmptyDocumentError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    @app.exception_handler(DocumentValidationError)
    async def document_validation_handler(
        request: Request,
        exc: DocumentValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    @app.exception_handler(DocumentPersistenceError)
    async def document_persistence_handler(
        request: Request,
        exc: DocumentPersistenceError,
    ) -> JSONResponse:
        logger.error("Database persistence failure: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Database persistence failure."},
        )

    @app.exception_handler(DocumentStorageError)
    async def document_storage_handler(
        request: Request,
        exc: DocumentStorageError,
    ) -> JSONResponse:
        logger.error("Document storage failure: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Document storage operation failed."},
        )

    @app.exception_handler(DocumentServiceError)
    async def document_service_handler(
        request: Request,
        exc: DocumentServiceError,
    ) -> JSONResponse:
        logger.error("Document service general error: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal service error occurred."},
        )

    @app.exception_handler(InvalidQueryError)
    async def invalid_query_handler(
        request: Request,
        exc: InvalidQueryError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    @app.exception_handler(RetrievalError)
    async def retrieval_error_handler(
        request: Request,
        exc: RetrievalError,
    ) -> JSONResponse:
        logger.error("Retrieval operation failure: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Retrieval operation failed."},
        )

    @app.exception_handler(GenerationError)
    async def generation_error_handler(
        request: Request,
        exc: GenerationError,
    ) -> JSONResponse:
        logger.error("Generation operation failure: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Answer generation failed."},
        )
