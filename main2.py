import os
import argparse
import datetime
import logging
import random

import torch
from torch.utils.data import Subset, TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
import torchmetrics
# from torchsummary import summary

import shap

from layers_custom2 import SparseTF, PNet, SimpleModel
from reactome import get_layer_maps
from dataset import MyDataset

# from interpret import shap_explainer

def main(args):
    """
    这个main中的模型修改后最终输出结果为每层输出结果平均值；
    并增加了shap归因方法进行解释
    """
    # 用当前时间作为文件夹名字，就不会跟前面的记录混乱了
    if args.log_dir == "default":
        args.log_dir = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(args.log_dir, exist_ok=True)

    # 设置打印日志，同时输出到终端和日志文件
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(os.path.join(args.log_dir, args.log))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)

    # 构建input和label
    combine = pd.read_csv(args.input)
    # random.seed = 123
    # random_numbers = random.sample(range(combine.shape[0]), 200)
    # combine = combine.iloc[random_numbers]
    X = torch.Tensor(combine.iloc[:, :-1].values)
    y = torch.Tensor(combine['prognosis'].values)
    logger.info(f"X shape: {X.shape}")
    # print(sum(y))


    # nan值用0代替
    X = torch.where(torch.isnan(X), torch.zeros_like(X), X)

    # 数据要不要归一化？ X 2% -> 0   98% -> 1
    X_2 = torch.quantile(X, 0.02, dim=0, keepdim=True)
    X_98 = torch.quantile(X, 0.98, dim=0, keepdim=True)
    X_2 = X_2.expand(X.shape[0], X.shape[1])
    X_98 = X_98.expand(X.shape[0], X.shape[1])
    X = torch.where((X_98 - X_2) < 1e-6, torch.zeros_like(X), (X - X_2) / (X_98 - X_2))
    X = torch.clamp(X, min=0, max=1)
    # X = torch.log()

    # 交叉验证集
    seed = 666

    # 5折交叉验证
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # 获取maps
    genes = combine.columns[:-1]
    maps = get_layer_maps(genes, n_levels=5, direction='root_to_leaf', add_unk_genes=True)

    # metrics，用sklearn完全没问题，只是我更习惯torchmetrics
    metrics = {
        "accuracy": torchmetrics.Accuracy(task="binary", threshold=0.5),
        "recall": torchmetrics.Recall(task="binary", threshold=0.5),
        "precision": torchmetrics.Precision(task="binary", threshold=0.5)
    }

    fold = 0
    for train, valid in kfold.split(X, y):
        fold += 1
        logger.info(f"Start Training fold: {fold}")

        # 设置tensorboard，用于可视化loss和metrics
        writer = SummaryWriter(os.path.join(args.log_dir, "fold_" + str(fold)))

        # 创建dataset
        train_dataset = MyDataset(X[train].clone(), y[train].clone())
        val_dataset = MyDataset(X[valid].clone(), y[valid].clone())

        # 创建训练和验证数据集
        dl_train = DataLoader(train_dataset, shuffle=True, batch_size=int(args.bs), drop_last=False)
        dl_val = DataLoader(val_dataset, shuffle=True, batch_size=int(args.bs), drop_last=False)

        # 构建模型，model里面包含了各种sparse, dense,...
        if args.model == 'PNet':
            model = PNet(maps)
        elif args.model == 'Simple':
            model = SimpleModel(maps)

        # 构建损失函数，reduction可以是none, mean, sum
        criterion = torch.nn.BCELoss(reduction="mean")

        # 构建优化器，需要传入模型的参数
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        model.to(args.device)
        global_step = 0
        for epoch in range(args.epochs):
            model.train()
            for step, (inputs, labels) in enumerate(dl_train):
                # 前向传播，实际调用的是model的forward()

                inputs = inputs.to(args.device)
                labels = labels.to(args.device)

                final_outcomes = model(inputs)
                global_step += 1

                # summary(model, input_size=inputs[0].size())
                # 计算损失
                loss_total = criterion(final_outcomes.squeeze(1), labels)
                writer.add_scalar(f"train/loss_total", loss_total, global_step)

                # 这三行一般都是这么写的
                # zero_grad()是把模型参数梯度清零
                # backward()是计算参数的梯度
                # step()是模型参数根据梯度走一步
                optimizer.zero_grad()
                loss_total.backward()
                # tmp = next(model.parameters())
                # logger.info(next(model.parameters())[0])
                # logger.info(next(model.parameters()).grad[0])
                optimizer.step()

            # 在验证集上进行验证
            model.eval()
            labels_total = []
            final_outcomes_total = []
            val_loss = 0

            # 解释
            if epoch == args.epochs - 1:
                shap.initjs()
                X = X.to(args.device)
                randint = np.random.choice(X[train].shape[0], 100, replace=False)
                explainer = shap.DeepExplainer(model, X[train][randint])
                shap_value = explainer.shap_values(X[valid][1:5])
                # shap_obj = explainer(X[valid])

                # shap_values.append(shap_value)
                # shap.force_plot(explainer.expected_value, shap_value, X[valid][1:5])
                # shap.plots.beeswarm(shap_obj)
                # shap_value = explainer.shap_values(inputs)

                # summarize the effects of all the features
                # shap.plots.waterfall(shap_value)
                shap.summary_plot(shap_values=shap_value,
                                  features=X[valid][1:5],
                                  feature_names=combine.columns
                                 # plot_type='waterfall'
                                  )

            with torch.no_grad():
                for val_step, (inputs, labels) in enumerate(dl_val):
                    inputs = inputs.to(args.device)
                    labels = labels.to(args.device)
                    final_outcomes = model(inputs)

                    # 计算每个batch的损失
                    loss = criterion(final_outcomes.squeeze(1), labels)
                    val_loss += loss.item()

                    final_outcomes_total.append(final_outcomes)
                    labels_total.append(labels)

            writer.add_scalar(f"val/loss", val_loss / len(dl_val), global_step)

            labels_total = torch.cat(labels_total, dim=0)
            final_outcomes_total = torch.cat(final_outcomes_total)

            # 更新每个batch指标
            for key, metric in metrics.items():
                metric = metric.to(args.device)
                value = metric(final_outcomes_total.squeeze(1), labels_total)
                writer.add_scalar(f"val/{key}", value, global_step)

        # 算每个epoch的指标值并输出
        for key, metric in metrics.items():
            computed_metric = metric.compute()
            logger.info(f'Epoch {epoch + 1}, Validation {key.capitalize()}: {computed_metric:.4f}')
            # 记录到 TensorBoard
            # writer.add_scalar(f"val/epoch_{key}", computed_metric, epoch+1)

        # 重置指标对象，以便下个 epoch 和 fold 计算
        for metric in metrics.values():
            metric.reset()

        logger.info(f'Finished Training Fold {fold}.')
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='Pca_combine_se.csv', help='选择combine文件名')
    parser.add_argument('--epochs', default=200, help='训练多少轮')
    parser.add_argument('--bs', default=8, help='batch size')
    parser.add_argument('--lr', default=1e-3, help='learning rate')
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        choices=['cuda', 'cpu'], help='训练设备，cpu或者gpu')
    parser.add_argument('--log', default='log.txt', help="保存的日志")
    parser.add_argument('--log-dir', default='default', help='日志文件夹')
    parser.add_argument('--model', default='PNet', choices=['PNet', 'Simple'])
    args = parser.parse_args()
    main(args)