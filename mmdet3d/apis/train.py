from mmdet.apis import train_detector


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """Train model.

    Args:
        model (nn.Module): Model to be tested.
        dataset (nn.Dataset): Dataset class.
        cfg (dict): Config
        distributed (bool): Whether to use distributed training
        validate (bool): whether to validate the model during training

    Returns:
        list[dict]: The prediction results.
    """
    train_detector(
        model,
        dataset,
        cfg,
        distributed=distributed,
        validate=validate,
        timestamp=timestamp,
        meta=meta)