

def feature_engineering(data_df):

    """
        Función para encapsular la tarea de ingeniería de variables

        Args:
           data_df (DataFrame):  Dataset de entrada.

        Returns:
           DataFrame. Datasets de salida.
    """

    data_df = create_domain_knowledge_features(data_df)

    return data_df.copy()


def create_domain_knowledge_features(data_df):

    """
        Función la creación de variables de contexto

        Args:
           df (DataFrame):  Dataset.
        Returns:
           DataFrame. Dataset.
    """

    data_df['Child'] = 0
    data_df.loc[data_df.Age < 16, 'Child'] = 1
    return data_df.copy()
