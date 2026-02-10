from utils_ import *;
from model import MyNet
import argparse
import copy
import os
import random
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default=3, help='3(mimic-iii) or 4(mimic-iv)')
parser.add_argument('--dim', type=int, default=256, help='embedding dimension')
parser.add_argument('--batch', type=int, default=32, help='mini-batch size')
parser.add_argument('--visit', type=int, default=3, help='history visit length')
parser.add_argument('--seed', type=int, default=1203, help='seed')
parser.add_argument('--test', type=int, default=0, help='test mode')
parser.add_argument('--save_model', type=int, default=0, help='save model (1=save best to --save_path)')
parser.add_argument('--save_path', type=str, default='weights/best_model.pth', help='path to save the best model when --save_model > 0')
parser.add_argument('--epoches', type=int, default=50, help='epoches')
# 预训练相关参数
parser.add_argument('--pretrain_mask', action='store_true', help='whether to use mask prediction pretrain')
parser.add_argument('--pretrain_epochs', type=int, default=1, help='number of pretrain epochs')
parser.add_argument('--pretrain_early_stop', type=int, default=5, help='early stop after this many epochs without improvement in pretrain')
parser.add_argument('--mask_prob', type=float, default=0.15, help='mask probability')
parser.add_argument('--pretrain_lr', type=float, default= 0.0001, help='learning rate for pretraining (if None, use --lr)')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate for finetuning')
# backbone / 消融相关参数
parser.add_argument('--backbone_mode', type=str, default='improved',
                    choices=['improved'],
                    help="backbone 模式: 'improved' 使用改进版(完整特征)")
# args = parser.parse_args(args=[])
args = parser.parse_args()


# load data
dataset=''
if args.dataset == 3:
    dataset = 'mimic-iii'
elif args.dataset == 4:
    dataset = 'mimic-iv'
data,voc,ddi_A,ehr_A=get_data('data/{}/'.format(dataset))
size_diag_voc, size_pro_voc, size_med_voc = len(voc['diag_voc']['idx2word']), len(voc['pro_voc']['idx2word']), len(voc['med_voc']['idx2word'])

voc_size=(size_diag_voc, size_pro_voc, size_med_voc)
print(f"Vocabulary sizes - Diag: {size_diag_voc}, Proc: {size_pro_voc}, Med: {size_med_voc}")
# data = [x for x in data if len(x) >= 5]  ### Only admissions more than twice as input ####
split_point = int(len(data) * 2 / 3)
data_train = data[:split_point]
eval_len = int(len(data[split_point:]) / 2)
data_test = data[split_point:split_point + eval_len]
data_eval = data[split_point+eval_len:]

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

med_freq=get_med_freq(data,size_med_voc).to(device)
ehr_A_gpu=torch.tensor(ehr_A).to(device)
ddi_A_gpu=torch.tensor(ddi_A).to(device)


model = MyNet(emb_dim=args.dim, k=args.visit, voc_size=voc_size,ehr_adj=ehr_A_gpu,ddi_adj=ddi_A_gpu).to(device)
# model.load_state_dict(torch.load('F:\code_2026\TPR-MR1\TPR-MR-main\weights\\best_model.pth'))

num_params = count_parameters(model)
print(f"Model parameters: {num_params:,}")
pretrain_lr = args.pretrain_lr if args.pretrain_lr is not None else args.lr
pretrain_weight_decay = 1e-4
optimizer = Adam(model.parameters(), lr=pretrain_lr, betas=(0.9, 0.999), eps=1e-05, weight_decay=pretrain_weight_decay)

# 显示预训练配置信息
print("预训练配置信息")
print(f"Mask预训练: {'启用' if args.pretrain_mask else '禁用'}")
if args.pretrain_mask:
    print(f"预训练轮数: {args.pretrain_epochs}")
    print(f"预训练学习率: {pretrain_lr}")
    print(f"Mask概率: {args.mask_prob}")
print(f"微调学习率: {args.lr}")
print("="*80 + "\n")

# (diag_batch,proc_batch,med_batch,label_batch,ids_batch,batch_seq_len)
infer_batch_size=256
train_batches=create_batches_dict(data_train, args.batch, voc_size, device)
eval_batches=create_batches_dict(data_eval, infer_batch_size, voc_size, device)

# shape: [batch, size_med_voc]
def loss_func(output, y_gt):
    # focal loss?
    return F.binary_cross_entropy_with_logits(output, y_gt)
    # bce_loss=F.binary_cross_entropy_with_logits(output, y_gt, reduction='none')
    # weights=1.0-med_freq
    # return (weights.view(1,-1).repeat(output.shape[0],1)*bce_loss).sum() * 10

def infer(model, batches, threshold):
    model.eval()
    with torch.no_grad():
        y_gt_full=[]
        y_pred_full=[]
        y_pred_prob_full=[]
        loss_full=0
        n_samples=0
        for batch_id in range(batches['n_batches']):
            output = model(
                batches['diag_batches'][batch_id],
                batches['proc_batches'][batch_id],
                batches['med_batches'][batch_id],
                mode='finetune'
            )
            
            # y_pred_prob: [batch, size_med_voc], the probability of each med label
            y_pred_prob=F.sigmoid(output)
            # y_pred: [batch, size_med_voc], the predicted med labels using threshold
            y_pred=(y_pred_prob >= threshold)
            y_gt=batches['label_batches'][batch_id]
            loss=loss_func(output,y_gt)
            y_gt_full.append(y_gt)
            y_pred_full.append(y_pred)
            y_pred_prob_full.append(y_pred_prob)
            loss_full+=loss.item() * y_gt.shape[0]
            n_samples+=y_gt.shape[0]
        return torch.cat(y_gt_full).cpu(), torch.cat(y_pred_full), torch.cat(y_pred_prob_full), loss_full/n_samples
# result=infer(model, eval_batches,threshold=0.3)
def infer_on_validation_data(model,threshold):
    return infer(model, eval_batches,threshold)

def get_metrics(model, batches, threshold=0.3):
    # restore the original order of samples to aggregate patient-level labels and predictions
    batches=eval_batches
    indexes=np.concatenate([np.array(a) for a in batches['ids_batches']])
    sorted_indexes=sorted([(i,indexes[i]) for i in range(len(indexes))], key=lambda x: x[1])
    o_indexes=[t[0] for t in sorted_indexes]
    p_len=[]
    counter=0
    for i in range(1,len(sorted_indexes)):
        if sorted_indexes[i][1] != sorted_indexes[i-1][1]:
            p_len.append(i-counter)
            counter=i
    p_len.append(len(sorted_indexes)-counter)

    infer_result=infer(model, batches, threshold)
    y_gt, y_pred,y_pred_prob,loss=infer_result
    y_gt=torch.stack([y_gt[i] for i in o_indexes],dim=0)
    y_pred=torch.stack([y_pred[i] for i in o_indexes],dim=0).cpu().numpy().astype(np.int32)
    y_pred_prob=torch.stack([y_pred_prob[i] for i in o_indexes],dim=0).cpu().numpy().astype(np.float64)

    # calculate scores in patient-level
    idx=0
    patient_scores=[]
    for len_visits in p_len:
        pred_meds=np.array(y_pred[idx:idx+len_visits,:])
        pred_meds_list=[np.where(pred_meds[i] == 1)[0].tolist() for i in range(pred_meds.shape[0])]
        result:list=list(multi_label_metric(
            np.array(y_gt[idx:idx+len_visits,:]),
            pred_meds,
            np.array(y_pred_prob[idx:idx+len_visits,:])
        ))
        result.append(ddi_rate_score(pred_meds_list, ddi_A))
        patient_scores.append(result)
        idx += len_visits
    patient_scores=np.array(patient_scores)
    ja,prauc,avg_precision,avg_recall,avg_f1,ddi=patient_scores.mean(axis=0)
    # average med num
    mean_med=y_pred.sum(axis=1).mean()
    return ja,prauc,avg_precision,avg_recall,avg_f1,ddi,mean_med,loss
# result=get_metrics(model, eval_batches, threshold=0.3)

def get_metrics_on_validation_data(model):
    return get_metrics(model, eval_batches)

def get_metrics_on_test_data(model):
    results=[]
    for i in range(10):
        random.shuffle(data_test)
        test_batches=create_batches_dict(data_test[:int(len(data_test)*0.8)], infer_batch_size, voc_size, device)
        result = list(get_metrics(model, test_batches))
        results.append(result)
    results=np.array(results)
    return results.mean(axis=0), results.std(axis=0)


def random_mask_word(seq, vocab_size, mask_prob=0.15):
   ...
  ...
...
    return masked_seq

def mask_batch_data(batch_data, mask_prob, voc_size):
    """
    对batch数据进行mask（按照RAREMed的逻辑）
    说明:
        - 只mask诊断和手术，不mask药物
        - 使用random_mask_word函数进行mask
    参数:
        batch_data: [[diag_list, proc_list], ...]
        mask_prob: 掩码概率
        voc_size: (d_voc_size, p_voc_size, m_voc_size)
    """
   ...
...
    return masked_data

def prepare_pretrain_data(data):
    """将数据转换为单visit格式用于预训练"""
    data_pretrain = []
    for patient in data:
        for visit in patient:
            data_pretrain.append(visit)
    return data_pretrain

def batchify_pretrain(data, batch_size):
    """将预训练数据分批"""
    batches = []
    for i in range(0, len(data), batch_size):
        batches.append(data[i:min(i + batch_size, len(data))])
    return batches

@torch.no_grad()
def evaluator_mask(model, data_val, voc_size, epoch, device):
    """评估Mask预训练任务"""
    model.eval()
    loss_val = 0
    loss_dis_val, loss_pro_val, loss_dis_med_val, loss_pro_med_val = 0, 0, 0, 0
    dis_ja_list, dis_prauc_list = [], []
    pro_ja_list, pro_prauc_list = [], []
    dis_med_ja_list, dis_med_prauc_list = [], []
    pro_med_ja_list, pro_med_prauc_list = [], []
    len_val = len(data_val)
    
    for batch in tqdm(data_val, ncols=60, desc="eval_mask", total=len_val):
        batch_size = len(batch)
        input_batch = [[visit[0], visit[1]] for visit in batch]
        
        # 将input_batch拆分成两个独立的列表
        diags_list = [item[0] for item in input_batch]
        procs_list = [item[1] for item in input_batch]
        # meds参数在pretrain_mask模式下不会被使用，但需要传递
        meds_list = [[] for _ in range(batch_size)]
        
        result = model(diags_list, procs_list, meds_list, mode='pretrain_mask')
        
        # 分离各个任务的预测结果
        result_dis = result[:, :voc_size[0]]
        result_pro = result[:, voc_size[0]:voc_size[0]+voc_size[1]]
        result_dis_med = result[:, voc_size[0]+voc_size[1]:voc_size[0]+voc_size[1]+voc_size[2]]
        result_pro_med = result[:, voc_size[0]+voc_size[1]+voc_size[2]:]
        
        # 目标
        dis_gt = np.zeros((batch_size, voc_size[0]))
        pro_gt = np.zeros((batch_size, voc_size[1]))
        med_gt = np.zeros((batch_size, voc_size[2]))
        for i in range(batch_size):
            dis_gt[i, batch[i][0]] = 1
            pro_gt[i, batch[i][1]] = 1
            if len(batch[i]) > 2:
                med_gt[i, batch[i][2]] = 1
        
        # 计算损失
        loss_dis = F.binary_cross_entropy_with_logits(
            result_dis, torch.tensor(dis_gt, device=device))
        loss_pro = F.binary_cross_entropy_with_logits(
            result_pro, torch.tensor(pro_gt, device=device))
        loss_dis_med = F.binary_cross_entropy_with_logits(
            result_dis_med, torch.tensor(med_gt, device=device))
        loss_pro_med = F.binary_cross_entropy_with_logits(
            result_pro_med, torch.tensor(med_gt, device=device))
        # loss = (loss_dis + loss_pro + loss_dis_med + loss_pro_med) / 4
        loss = ( loss_dis_med + loss_pro_med) / 2
        # loss = (loss_dis + loss_pro) / 2

        loss_val += loss.item()
        loss_dis_val += loss_dis.item()
        loss_pro_val += loss_pro.item()
        loss_dis_med_val += loss_dis_med.item()
        loss_pro_med_val += loss_pro_med.item()
        
        # 评估
        dis_pred_prob = F.sigmoid(result_dis).cpu().numpy()
        pro_pred_prob = F.sigmoid(result_pro).cpu().numpy()
        dis_med_pred_prob = F.sigmoid(result_dis_med).cpu().numpy()
        pro_med_pred_prob = F.sigmoid(result_pro_med).cpu().numpy()
        
        dis_pred = (dis_pred_prob >= 0.5).astype(int)
        pro_pred = (pro_pred_prob >= 0.5).astype(int)
        dis_med_pred = (dis_med_pred_prob >= 0.5).astype(int)
        pro_med_pred = (pro_med_pred_prob >= 0.5).astype(int)
        
        dis_ja, dis_prauc, _, _, _ = multi_label_metric(dis_gt, dis_pred, dis_pred_prob)
        pro_ja, pro_prauc, _, _, _ = multi_label_metric(pro_gt, pro_pred, pro_pred_prob)
        dis_med_ja, dis_med_prauc, _, _, _ = multi_label_metric(med_gt, dis_med_pred, dis_med_pred_prob)
        pro_med_ja, pro_med_prauc, _, _, _ = multi_label_metric(med_gt, pro_med_pred, pro_med_pred_prob)
        
        dis_ja_list.append(dis_ja)
        dis_prauc_list.append(dis_prauc)
        pro_ja_list.append(pro_ja)
        pro_prauc_list.append(pro_prauc)
        dis_med_ja_list.append(dis_med_ja)
        dis_med_prauc_list.append(dis_med_prauc)
        pro_med_ja_list.append(pro_med_ja)
        pro_med_prauc_list.append(pro_med_prauc)
    
    # 只使用“诊断->药物”和“手术->药物”的Jaccard来评估最佳模型
    # 因为训练时只优化了 loss_dis_med 和 loss_pro_med，需要让训练目标与评估目标一致
    ja = (np.mean(dis_med_ja_list) + np.mean(pro_med_ja_list)) / 2  # 只计算药物相关任务的平均Jaccard
    # avg_ja 仍然保留四个任务的平均Jaccard，用于日志展示
    avg_ja = (np.mean(dis_ja_list) + np.mean(pro_ja_list) + np.mean(dis_med_ja_list) + np.mean(pro_med_ja_list)) / 4
    avg_prauc = (np.mean(dis_prauc_list) + np.mean(pro_prauc_list) + np.mean(dis_med_prauc_list) + np.mean(pro_med_prauc_list)) / 4
    
    metrics = {
        'loss_val': loss_val/len_val,
        'loss_dis': loss_dis_val/len_val,
        'loss_pro': loss_pro_val/len_val,
        'loss_dis_med': loss_dis_med_val/len_val,
        'loss_pro_med': loss_pro_med_val/len_val,
        'dis_ja': np.mean(dis_ja_list),
        'dis_prauc': np.mean(dis_prauc_list),
        'pro_ja': np.mean(pro_ja_list),
        'pro_prauc': np.mean(pro_prauc_list),
        'dis_med_ja': np.mean(dis_med_ja_list),
        'dis_med_prauc': np.mean(dis_med_prauc_list),
        'pro_med_ja': np.mean(pro_med_ja_list),
        'pro_med_prauc': np.mean(pro_med_prauc_list),
        'ja': ja,  # 这里的 ja 已经改为“诊断->药物 + 手术->药物”的平均Jaccard（用于选择最佳模型）
        'avg_ja': avg_ja,  # 所有任务的平均Jaccard（用于显示）
        'avg_prauc': avg_prauc
    }
    
    return metrics

def main_mask(args, model, optimizer, data_train, data_val, voc_size, device):
    """Mask预训练主函数"""
    
    epoch_mask = 0
    best_ja_mask = -1  # 初始化为-1，避免第一轮任何值都成为最佳
    best_epoch_mask = 0  # 记录最佳轮次
    no_improve_count_mask = 0  # 连续未提升的轮次数
    EPOCH = args.pretrain_epochs
    
    log_file = create_log_file('pretrain_mask')
    log(log_file, "Mask预训练开始...")
    log(log_file, f"预训练轮数: {EPOCH}, Mask概率: {args.mask_prob}")
    log(log_file, 'epoch\tloss_train\tloss_dis\tloss_pro\tloss_dis_med\tloss_pro_med\tloss_val\tloss_dis_val\tloss_pro_val\tloss_dis_med_val\tloss_pro_med_val')
    log(log_file, 'epoch\tdis_ja\tdis_prauc\tpro_ja\tpro_prauc\tdis_med_ja\tdis_med_prauc\tpro_med_ja\tpro_med_prauc\tavg_ja\tavg_prauc\tbest_ja')
    
    for epoch in range(EPOCH):
        epoch += 1
        model.train()
        epoch_mask += 1
        loss_train = 0
        loss_dis_train, loss_pro_train, loss_dis_med_train, loss_pro_med_train = 0, 0, 0, 0
        for batch in tqdm(data_train, ncols=60, desc=f"Mask Pretrain Epoch {epoch}", total=len(data_train)):
            batch_size = len(batch)
            input_batch = [[visit[0], visit[1]] for visit in batch]
            
            if args.mask_prob > 0:
                masked_batch = mask_batch_data(input_batch, args.mask_prob, voc_size)
            else:
                masked_batch = input_batch
            
            # 将masked_batch拆分成两个独立的列表
            diags_list = [item[0] for item in masked_batch]
            procs_list = [item[1] for item in masked_batch]
            # meds参数在pretrain_mask模式下不会被使用，但需要传递
            meds_list = [[] for _ in range(batch_size)]
            
            result = model(diags_list, procs_list, meds_list, mode='pretrain_mask')
            
            # 分离各个任务的预测结果
            result_dis = result[:, :voc_size[0]]
            result_pro = result[:, voc_size[0]:voc_size[0]+voc_size[1]]
            result_dis_med = result[:, voc_size[0]+voc_size[1]:voc_size[0]+voc_size[1]+voc_size[2]]
            result_pro_med = result[:, voc_size[0]+voc_size[1]+voc_size[2]:voc_size[0]+voc_size[1]+voc_size[2]*2]
            
            # 目标
            dis_gt = np.zeros((batch_size, voc_size[0]))
            pro_gt = np.zeros((batch_size, voc_size[1]))
            med_gt = np.zeros((batch_size, voc_size[2]))
            for i in range(batch_size):
                dis_gt[i, batch[i][0]] = 1
                pro_gt[i, batch[i][1]] = 1
                if len(batch[i]) > 2:
                    med_gt[i, batch[i][2]] = 1
            
            # 计算ATC3级损失
            loss_dis = F.binary_cross_entropy_with_logits(
                result_dis, torch.tensor(dis_gt, device=device))
            loss_pro = F.binary_cross_entropy_with_logits(
                result_pro, torch.tensor(pro_gt, device=device))
            loss_dis_med = F.binary_cross_entropy_with_logits(
                result_dis_med, torch.tensor(med_gt, device=device))
            loss_pro_med = F.binary_cross_entropy_with_logits(
                result_pro_med, torch.tensor(med_gt, device=device))
            loss = (loss_dis_med + loss_pro_med) / 2
            # loss = (loss_dis + loss_pro ) / 2

            # loss = (loss_dis + loss_pro + loss_dis_med + loss_pro_med) / 4
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_train += loss.item()
            loss_dis_train += loss_dis.item()
            loss_pro_train += loss_pro.item()
            loss_dis_med_train += loss_dis_med.item()
            loss_pro_med_train += loss_pro_med.item()
        
        loss_train /= len(data_train)
        loss_dis_train /= len(data_train)
        loss_pro_train /= len(data_train)
        loss_dis_med_train /= len(data_train)
        loss_pro_med_train /= len(data_train)
        
        # validation
        print('Validating...')
        sys.stdout.flush()
        metrics = evaluator_mask(model, data_val, voc_size, epoch, device)
        
        is_best = False

        if best_ja_mask < 0:  # 第一轮，初始化
            print(f'\n📊 第一轮评估: Jaccard = {metrics["ja"]:.4f}')
            best_ja_mask = metrics['ja']
            best_epoch_mask = epoch
            is_best = True
        elif metrics['ja'] > best_ja_mask:
            old_best = best_ja_mask
            best_ja_mask = metrics['ja']
            best_epoch_mask = epoch  # 更新最佳轮次
            no_improve_count_mask = 0  # 重置未提升计数
            is_best = True
            print(f'[Mask Pretrain Epoch {epoch}/{EPOCH}] Best Jaccard: {old_best:.4f} -> {metrics["ja"]:.4f}')
        else:
            no_improve_count_mask += 1
            if no_improve_count_mask >= args.pretrain_early_stop:
                print(f'[Mask Pretrain] Early stop at epoch {epoch}, Best Jaccard: {best_ja_mask:.4f} (epoch {best_epoch_mask})')
                break
        
        if is_best:
            print(f'[Mask Pretrain Epoch {epoch}/{EPOCH}] Best Jaccard: {metrics["ja"]:.4f} (Best epoch: {best_epoch_mask})')
        else:
            print(f'[Mask Pretrain Epoch {epoch}/{EPOCH}] Jaccard: {metrics["ja"]:.4f} (Best: {best_ja_mask:.4f} @ epoch {best_epoch_mask})')
        
        # 写入日志
        log(log_file, f'{epoch}\t{loss_train:.4f}\t{loss_dis_train:.4f}\t{loss_pro_train:.4f}\t{loss_dis_med_train:.4f}\t{loss_pro_med_train:.4f}\t{metrics["loss_val"]:.4f}\t{metrics["loss_dis"]:.4f}\t{metrics["loss_pro"]:.4f}\t{metrics["loss_dis_med"]:.4f}\t{metrics["loss_pro_med"]:.4f}')
        log(log_file, f'{epoch}\t{metrics["dis_ja"]:.4f}\t{metrics["dis_prauc"]:.4f}\t{metrics["pro_ja"]:.4f}\t{metrics["pro_prauc"]:.4f}\t{metrics["dis_med_ja"]:.4f}\t{metrics["dis_med_prauc"]:.4f}\t{metrics["pro_med_ja"]:.4f}\t{metrics["pro_med_prauc"]:.4f}\t{metrics["avg_ja"]:.4f}\t{metrics["avg_prauc"]:.4f}\t{best_ja_mask:.4f}')
    
    log(log_file, f'\nMask预训练完成，最佳Jaccard: {best_ja_mask:.4f} (第 {best_epoch_mask} 轮)')
    print(f'\nMask预训练完成，最佳Jaccard: {best_ja_mask:.4f} (第 {best_epoch_mask} 轮)')
    log_file.close()
    return model

def train(model, optimizer):

    log_file=create_log_file('train_mynet')
    log(log_file, 'config: ' + str(args) + '\n')
    log(log_file, "traing...")
    log(log_file, time.strftime("Date: %Y%m%d-%H%M%S"))
    log(log_file, "File: {}".format(__file__) + '\n')
    log(log_file, 'params: %d' % count_parameters(model))
    log(log_file, 'epoch\tja\tprauc\tavg_p\tavg_r\tavg_f1\tddi\tavg_med\tt_loss\tv_loss')

    start_time=time.time()
    continuous_decline=0
    prev_prauc = 0.0
    best=[0]*100
    # 步骤5.1: 保存当前模型状态（如果进行了预训练，这里保存的是预训练后的状态）
    best_model_params=model.state_dict()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    for epoch in range(args.epoches):
        train_loss_sum=0
        n_samples=0
        model_train = True
        if model_train:
            for i in tqdm(range(train_batches['n_batches']), ncols=60, desc=f"Fine-tune Epoch {epoch+1}/{args.epoches}"):
                blen=train_batches['diag_batches'][i].shape[0]
                n_samples+=blen
                output = model(
                    train_batches['diag_batches'][i],
                    train_batches['proc_batches'][i],
                    train_batches['med_batches'][i],
                    mode='finetune'
                )
                loss = loss_func(output, train_batches['label_batches'][i])
                train_loss_sum += loss.detach().item() * blen
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        # check performance
        ja,prauc,avg_precision,avg_recall,avg_f1,ddi,mean_med,v_loss=get_metrics_on_validation_data(model)
        t_loss=train_loss_sum/n_samples
        metric=(epoch+1,ja,prauc,avg_precision,avg_recall,avg_f1,ddi,mean_med,t_loss,v_loss)
        log(log_file, '%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.2f\t%.4f\t%.4f' % metric)
        # save the best model
        if prauc > best[2]:
            best = metric
            best_model_params = copy.deepcopy(model.state_dict())
        # early stop
        if prauc < prev_prauc:
            continuous_decline += 1
            if continuous_decline >= 2:
                log(log_file, ' [ early stop ] ')
                break
        else:
            continuous_decline = 0
        prev_prauc = prauc
    log(log_file, 'best:')
    log(log_file, '%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.2f\t%.4f\t%.4f' % best)
    log(log_file, "training stopped.")
    log(log_file, "Time used: %.2f" % (time.time()-start_time))
    log_file.close()
    # end training
    return best_model_params


# 预训练阶段
sys.stdout.flush()

try:
    if args.pretrain_mask:
        # 准备预训练数据
        try:
            data_train_pretrain = prepare_pretrain_data(data_train)
            data_val_pretrain = prepare_pretrain_data(data_eval)
            data_train_pretrain_batches = batchify_pretrain(data_train_pretrain, args.batch)
            data_val_pretrain_batches = batchify_pretrain(data_val_pretrain, args.batch)
        except Exception as e:
            print(f"Error preparing pretrain data: {e}")
            raise
        
        # Mask预训练
        try:
            model = main_mask(args, model, optimizer, data_train_pretrain_batches, data_val_pretrain_batches, voc_size, device)
        except Exception as e:
            print(f"Error in Mask pretraining: {e}")
            raise
        
        optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-05, weight_decay=1e-06)
except Exception as e:
    print(f"Error in pretraining stage: {e}")
    raise

# 注意: model对象在预训练阶段已经被更新，包含预训练权重
best_model_params = train(model, optimizer)
if args.save_model > 0:
    save_path = args.save_path
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(best_model_params, save_path)
    print(f'Best model saved to {save_path}')
if args.test > 0:
    model.load_state_dict(best_model_params)
    result=get_metrics_on_test_data(model)
    print(result)
