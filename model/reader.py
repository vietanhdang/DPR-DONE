from haystack.reader.farm import FARMReader


def read():
    reader = FARMReader(model_name_or_path='deepset/roberta-base-squad2'
                        , progress_bar=False, use_gpu=False,
                        num_processes=4)

    return reader
