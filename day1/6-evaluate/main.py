'''
    Evaluate Model
    Release Date: 2025-03-02
'''


#=====================#
# ---- Libraries ---- #
#=====================#
import os
import argparse
import logging
import pandas as pd
import wandb
import mlflow.sklearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

# === NUEVAS IMPORTACIONES SI NECESITAS RECREAR EL PREPROCESADOR AQUÍ ===
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# =======================================================================

#==========================#
#   Logger Configuration   #
#==========================#
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

#=========================#
# ---- Main Function ---- #
#=========================#

def go(args):
    run = wandb.init(job_type="test")

    logger.info("Downloading and reading test artifact")
    test_data_path = run.use_artifact(args.test_data).file()
    df = pd.read_csv(test_data_path,
                     low_memory=False)

    logger.info("Extracting target from dataframe")
    X_test = df.copy()
    y_test = X_test.pop("median_house_value")

    logger.info("Downloading and reading the exported model")
    model_download_root_path = run.use_artifact(args.model_export).download()

    # Cargar el RandomForestRegressor directamente
    loaded_random_forest_model = mlflow.sklearn.load_model(model_download_root_path)


    logger.info("Transform data with Pipeline")
    # AHORA EL PREPROCESADOR NO ES PARTE DEL MODELO CARGADO.
    # Debes recrear el preprocesador con la misma lógica que usaste en el entrenamiento
    # y aplicarlo a X_test ANTES de pasar los datos al RandomForestRegressor.

    # Ejemplo de cómo recrear el preprocesador (AJUSTA ESTO SEGÚN TU PREPROCESAMIENTO REAL)
    # Necesitas saber qué columnas son numéricas y categóricas, y qué transformadores aplicaste.
    numeric_features = X_test.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_test.select_dtypes(include='object').columns.tolist()

    # Recrear el preprocesador. Es CRUCIAL que este preprocesador sea idéntico
    # al que se usó en el entrenamiento (en términos de `fit` si no lo guardaste).
    # Si no guardaste el preprocesador, esta es una fuente de errores potenciales.
    # Idealmente, deberías haber guardado y logueado el preprocesador por separado
    # o como parte de la Pipeline.
    recreated_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough' # O 'drop' si descartaste las no especificadas
    )

    # Es importante que el preprocesador se ajuste (fit) con los datos de entrenamiento
    # o que uses un preprocesador que fue entrenado y guardado.
    # Dado que no lo cargamos del modelo, tendremos que "simular" su entrenamiento
    # para que las transformaciones (como StandardScaler) se hagan correctamente.
    # ¡ESTO ES PELIGROSO si los datos de test no son representativos para el fit!
    # La mejor práctica es que el preprocesador esté ya "fit" o sea parte de una Pipeline.

    # Si tu preprocesador no requiere 'fit' en tiempo de inferencia (e.g. solo OneHotEncoder con handle_unknown),
    # o si está "fit" con datos del entrenamiento, puedes usar .transform() directamente.
    # Para ser seguro, y si el preprocesador no se guardó, puedes hacer un fit_transform EN LOS DATOS DE ENTRENAMIENTO,
    # y luego solo transform en los datos de test. Pero eso implica tener los datos de entrenamiento disponibles aquí.

    # Asumiendo que el preprocesador se puede inicializar y usar para transformar directamente los datos de test
    # (lo cual es menos común para StandardScaler a menos que se hayan guardado sus parámetros).
    # Si tus transformadores requieren .fit(), es *fundamental* que los cargues ya `fit`
    # o que uses la solución de la Pipeline completa.
    X_test_processed = recreated_preprocessor.fit_transform(X_test) # O .transform() si el preprocesador ya está "fit" con datos de entrenamiento.


    logger.info("Making predictions")
    y_pred = loaded_random_forest_model.predict(X_test_processed) # Usa el modelo cargado directamente

    logger.info("Scoring (RMSE Calculation)")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    run.summary["RMSE"] = rmse
    logger.info(f"RMSE on test set: {rmse:.2f}")

    # ========================= #
    #   Visualization Plots     #
    # ========================= #

    logger.info("Generating Prediction vs Actual Values Scatter Plot")
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
    ax_scatter.scatter(y_test, y_pred, alpha=0.5, label="Predictions")
    ax_scatter.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Ideal Prediction")
    ax_scatter.set_xlabel("Actual Values")
    ax_scatter.set_ylabel("Predicted Values")
    ax_scatter.set_title("Prediction vs Actual Values")
    ax_scatter.legend()
    fig_scatter.tight_layout()

    logger.info("Generating Residuals Plot")
    fig_residuals, ax_residuals = plt.subplots(figsize=(10, 6))
    residuals = y_test - y_pred
    ax_residuals.scatter(y_test, residuals, alpha=0.5)
    ax_residuals.axhline(y=0, color="r", linestyle="--")
    ax_residuals.set_xlabel("Actual Values")
    ax_residuals.set_ylabel("Residuals")
    ax_residuals.set_title("Residuals Plot")
    fig_residuals.tight_layout()

    # Log plots in W&B
    run.log({
        "prediction_vs_actual": wandb.Image(fig_scatter),
        "residuals_plot": wandb.Image(fig_residuals)
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the provided Random Forest regression model",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--model_export",
        type=str,
        help="Fully-qualified artifact name for the exported model to evaluate",
        required=True,
    )

    parser.add_argument(
        "--test_data",
        type=str,
        help="Fully-qualified artifact name for the test data",
        required=True,
    )

    args = parser.parse_args()

    go(args)