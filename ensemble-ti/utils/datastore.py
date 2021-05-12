import ast
import datasets


def get_dataset(name, root, **kwargs):
    kwargs = _eval_kwargs(kwargs)
    if name == "mutation":
        dataset = datasets.MutationDataset(root, **kwargs)
    elif name == "deepamr":
        dataset = datasets.DeepAMRDataset(root, **kwargs)
    elif name == "rifampicin":
        dataset = datasets.RifampicinDataset(root, **kwargs)
    return dataset, dataset.num_classes


def _eval_kwargs(kwargs):
    for k, v in kwargs.items():
        try:
            kwargs[k] = ast.literal_eval(v)
        except ValueError:
            # A value error is thrown if the value is a str.
            continue
    return kwargs
