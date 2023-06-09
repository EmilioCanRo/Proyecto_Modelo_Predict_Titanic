from ..data.make_dataset import make_dataset
from app import cos, client
from cloudant.query import Query


def predict_pipeline(data, model_info_db_name='models-db'):

    """
        Función para gestionar el pipeline completo de inferencia
        del modelo.

        Args:
            path (str):  Ruta hacia los datos.

        Kwargs:
            model_info_db_name (str):  base de datos a usar para almacenar
            la info del modelo.

        Returns:
            list. Lista con las predicciones hechas.
    """

    # Carga de la configuración de entrenamiento
    model_config = load_model_config(model_info_db_name)['model_config']
    # columnas a retirar
    cols_to_remove = model_config['cols_to_remove']

    # obteniendo la información del modelo en producción
    model_info = get_best_model_info(model_info_db_name)
    # cargando y transformando los datos de entrada
    data_df = make_dataset(data, model_info, cols_to_remove)

    # Descargando el objeto del modelo
    model_name = model_info['name']+'.pkl'
    print('------> Loading the model {} object from the cloud'.format(model_name))
    model = load_model(model_name)

    # realizando la inferencia con los datos de entrada
    return model.predict(data_df).tolist()


def load_model(name, bucket_name='models-uem'):
    """
         Función para cargar el modelo en IBM COS

         Args:
             name (str):  Nombre de objeto en COS a cargar.

         Kwargs:
             bucket_name (str):  depósito de IBM COS a usar.

        Returns:
            obj. Objeto descargado.
     """
    return cos.get_object_in_cos(name, bucket_name)


def get_best_model_info(db_name):
    """
         Función para cargar la info del modelo de IBM Cloudant

         Args:
             db_name (str):  base de datos a usar.

         Kwargs:
             bucket_name (str):  depósito de IBM COS a usar.

        Returns:
            dict. Info del modelo.
     """
    db = client.get_database(db_name)
    query = Query(db, selector={'status': {'$eq': 'in_production'}})
    return query()['docs'][0]


def load_model_config(db_name):
    """
        Función para cargar la info del modelo desde IBM Cloudant.

        Args:
            db_name (str):  Nombre de la base de datos.

        Returns:
            dict. Documento con la configuración del modelo.
    """
    db = client.get_database(db_name)
    query = Query(db, selector={'_id': {'$eq': 'model_config'}})
    return query()['docs'][0]
