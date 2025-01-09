import mmcv
import torch

def single_gpu_test(model,
                    data_loader,
                    out_dir=None,
                    show_pretrain=False):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    torch.manual_seed(0)
    model.eval()

    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    
    
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, pretrain=show_pretrain, show_pretrain = True, out_dir=out_dir, **data)

        results.extend(result)

        batch_size = 1 
        for _ in range(batch_size):
            prog_bar.update()


    metrics = {}
    for key in results[0]:
        metrics[key] = []
    
    for result in results: 
        for key in result:
            metrics[key].append(result[key])
    
    print()
    for key in metrics.keys():
        if key == "ious":
            
            results = torch.cat(metrics[key], dim=0).mean(0)
            mean_ious = []
            class_num = len(model.module.class_names) + 1
            class_names = ["iou"]
            class_names.extend(model.module.class_names)
            
            for i in range(class_num):
                tp = results[i, 0]
                p = results[i, 1]
                g = results[i, 2]
                union = p + g - tp
                mean_ious.append(tp / union)
            
            for i in range(class_num):
                print("{}: {}".format(class_names[i], mean_ious[i]))
            print("{}: {}".format("mIoU", nanmean(torch.tensor(mean_ious[1:]))))
        else:
            print("{}: {}".format(key, torch.stack(metrics[key]).mean().item()))


    return metrics

def nanmean(tensor):
    tensor = tensor[~tensor.isnan()]
    return tensor.mean()