#!/usr/bin/env python
'''
    FastAPI Application for Post-COVID Model
    Release Date: 2025-01-27
'''

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loggers_configuration import setup_colored_logger
from routers import predictions

# Logger configuration
logger = setup_colored_logger("FastAPI-PostCOVID", "INFO")

# Initialize FastAPI app
app = FastAPI(
    title="Post-COVID Predictions API",
    description="API for predicting Post-COVID conditions using trained Random Forest model",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predictions.router, prefix="/api/v1", tags=["predictions"])

@app.get("/")
def root():
    """Root endpoint"""
    logger.info("Root endpoint accessed")
    return {
        "message": "Post-COVID Predictions API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    logger.info("Health check endpoint accessed")
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
