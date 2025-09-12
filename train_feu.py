import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
import os
import numpy as np
from network import *
import torch.nn.functional as F
from datasets.flicker import get_flicker1K_dataset, get_flicker_dataset
from datasets.visual_genome import get_VG_dataset_pc, get_VGtest_dataset
from datasets.coco import get_coco_dataset
from utils import interpret_batch, interpret_new, interpret, interpret_afa
import CLIP.clip as clip
from inference_grounding import inference_bbox
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def truncate_text(text, max_words=15):
    if not isinstance(text, str):
        raise ValueError(f"Expected string input, got {type(text)}")
    words = text.split()
    return ' '.join(words[:max_words])

def interpolate_image(image, size, mode="bilinear", align_corners=True):
    try:
        return F.interpolate(image, size=size, mode=mode, align_corners=align_corners)
    except Exception as e:
        logger.error(f"Image interpolation failed: {e}")
        raise

class AttentionContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(AttentionContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, F_f, A_pos, A_neg):
        """
        Args:
            F_f: (bs, d), fusion feature (anchor)
            A_pos: (bs, d), positive attention map
            A_neg: (bs, num_neg, d), negative attention maps

        Returns:
            loss: scalar
        """
        bs, d = F_f.shape
        num_neg = A_neg.shape[1]
        F_f = F_f / F_f.norm(dim=1, keepdim=True)
        A_pos = A_pos / A_pos.norm(dim=1, keepdim=True)
        A_neg = A_neg / A_neg.norm(dim=-1, keepdim=True)
        pos_sim = torch.sum(F_f * A_pos, dim=1, keepdim=True) / self.temperature
        neg_sim = torch.bmm(F_f.unsqueeze(1), A_neg.transpose(1, 2)).squeeze(1) / self.temperature
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(bs, dtype=torch.long, device=F_f.device)
        loss = F.cross_entropy(logits, labels)
        return loss

def norm_z(z):
    return z / z.norm(dim=1).unsqueeze(dim=1)

def open_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    a = os.listdir(path)
    os.mkdir(path + '/gpu' + str(len(a)))
    return str(len(a))

def get_logits(clip_model, real_imgs, text_pos, text_neg):
    logits_pos, _ = clip_model(real_imgs, text_pos)
    logits_neg, _ = clip_model(real_imgs, text_neg)
    logits_fr = torch.cat((logits_pos.diag().unsqueeze(-1),
                           logits_neg.diag().unsqueeze(-1)),
                          dim=1)
    return logits_fr

def gen_step(optimizer_G, clip_model, real_imgs, text, para, counter,  model, criterion, args):
    bs = real_imgs.shape[0]
    optimizer_G.zero_grad()
    device = real_imgs.device
    clip_model.to(device)
    model.to(device)
    text_pos = text[:, :, 0]
    z_t = norm_z(clip_model.encode_text(text_pos))
    para_pos = para[:, :, 0]
    z_p = norm_z(clip_model.encode_text(para_pos))
    counter_pos = counter[:, :, 0]
    z_c = norm_z(clip_model.encode_text(counter_pos))
    real_imgs_224 = F.interpolate(real_imgs, size=(224, 224), mode="bilinear", align_corners=True)
    cam = interpret_new(real_imgs_224.detach(), text_pos.detach(), clip_model, device).detach().float()
    cam = F.interpolate(cam, size=(int(args['Isize']), int(args['Isize'])), mode="bilinear", align_corners=True)
    M, afa, heatmap = model(real_imgs, z_t, text_pos, clip_model)
    heatmap = F.interpolate(heatmap, size=(int(args['Isize']), int(args['Isize'])), mode="bilinear", align_corners=True)
    M_c, afa_c, heatmap_c = model(real_imgs, z_c, counter_pos, clip_model)
    heatmap_c = F.interpolate(heatmap_c, size=(int(args['Isize']), int(args['Isize'])), mode="bilinear", align_corners=True)
    H = heatmap.sum(1).reshape(bs, -1)
    H_c = heatmap_c.sum(1).reshape(bs, 1, -1)
    C = cam.sum(1).reshape(bs, -1)
    criterion_contrast = AttentionContrastiveLoss(temperature=0.1).to(device)
    cam_loss = criterion_contrast(C, H, H_c)
    clip_cam_loss = F.mse_loss(M, cam)
    regularization = M.mean()
    M_up = F.interpolate(M, size=(224, 224), mode="bilinear", align_corners=True)
    z_fr = norm_z(clip_model.encode_image(real_imgs_224 * M_up))
    z_bg = norm_z(clip_model.encode_image(real_imgs_224 * (1 - M_up)))
    z_t_prime = torch.cat([z_t, z_p], dim=0)
    z_fr_rep = z_fr.repeat(2, 1)
    z_bg_rep = z_bg.repeat(2, 1)
    H_g_prime = torch.stack([z_fr_rep, z_bg_rep], dim=1)
    similarity = torch.matmul(z_t_prime.unsqueeze(1), H_g_prime.transpose(1, 2)) / 0.2
    similarity = similarity.squeeze(1)
    labels_contrast = torch.zeros(z_t_prime.size(0), dtype=torch.long).to(device)
    rc_loss = F.cross_entropy(similarity, labels_contrast)
    labels_pos = torch.ones(bs, dtype=torch.long).to(device)
    labels_neg = torch.zeros(bs, dtype=torch.long).to(device)
    L_ce_pos = F.cross_entropy(afa, labels_pos)
    L_ce_neg = F.cross_entropy(afa_c, labels_neg)
    L_ce = (L_ce_pos + L_ce_neg) * 0.5
    loss = (float(args['w1']) * clip_cam_loss +
            float(args['w0']) * regularization +
            float(args['w2']) * rc_loss +
            float(args['w4']) * cam_loss +
            float(args['w2']) * L_ce)

    loss.backward()
    optimizer_G.step()

    return loss.item(), L_ce.item()

def train(ds, model, clip_model, optimizer_G, args):
    loss_list = []
    ce_loss_list = []
    pbar = tqdm(ds)
    criterion = nn.CrossEntropyLoss()
    for i, inputs in enumerate(pbar):
        real_imgs = inputs[0].cuda()
        text_pos = inputs[1]
        para_pos = inputs[2]
        counter_pos = inputs[3]
        def truncate_text(text, max_words=15):
            words = text.split()
            return ' '.join(words[:max_words])
        para_pos_truncated = [truncate_text(p) for p in para_pos]
        counter_pos_truncated = [truncate_text(p) for p in counter_pos]
        text_pos_token = clip.tokenize(text_pos).to('cuda').unsqueeze(dim=2)
        para_pos_token = clip.tokenize(para_pos_truncated).to('cuda').unsqueeze(dim=2)
        counter_pos_token = clip.tokenize(counter_pos_truncated).to('cuda').unsqueeze(dim=2)
        loss, ce_loss = gen_step(
            optimizer_G, clip_model, real_imgs,
            text_pos_token, para_pos_token, counter_pos_token,
            model, criterion, args
        )
        loss_list.append(loss)
        ce_loss_list.append(ce_loss)
        pbar.set_description(
            '(train) :: loss {loss:.4f} | ce {ce:.4f}'.format(
                loss=np.mean(loss_list), ce=np.mean(ce_loss_list)
            )
        )
    return np.mean(loss_list), np.mean(ce_loss_list)

def logger(writer, loss_list, tplt_loss, step):
    writer.add_scalar('Loss', loss_list, global_step=step)
    writer.add_scalar('tplt_loss', tplt_loss, global_step=step)

def main(args=None):
    args['is_blip'] = False
    gpu_num = torch.cuda.device_count()
    model = MultiModel(args=args)
    model = torch.nn.DataParallel(model, list(range(gpu_num))).cuda()
    if bool(int(args['resume'])):
        model2 = torch.load(args['resume_folder'])
        print(int(args['resume']))
        model_dict = model.state_dict()
        state_dict = model2.state_dict()

        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        if len(filtered_dict) == 0:
            raise RuntimeError(f"No matching keys found in checkpoint {args['resume_folder']}")

        print(f" Successfully loaded layers:")
        for k in filtered_dict.keys():
            print(f"  {k}")
        print(f"Skipped (new layers, will be randomly initialized):")
        for k in model_dict.keys():
            if k not in filtered_dict:
                print(f"  {k}")
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        print(f" Resumed from epoch {args['resume']}, new layers (fc1/fc2) are randomly initialized.")
    else:
        print("Training from scratch.")
    optimizer_G = optim.SGD(model.parameters(),
                            lr=float(args['learning_rate']),
                            weight_decay=float(args['WD']),
                            momentum=float(args['M']))

    if args['task'] == 'flicker':
        trainset, testset = get_flicker_dataset(args=args)
    elif args['task'] == 'vg_train':
        trainset = get_VG_dataset_pc(args=args)
        testset = get_flicker1K_dataset(args=args)
    elif args['task'] == 'coco':
        trainset = get_coco_dataset(args=args)
        testset = get_flicker1K_dataset(args=args)
    elif args['task'] == 'vg_self':
        trainset = get_VG_dataset_pc(args=args)
        testset = get_VGtest_dataset(args)
    ds = torch.utils.data.DataLoader(trainset,
                                     batch_size=int(args['Batch_size']),
                                     num_workers=int(args['nW']),
                                     shuffle=True,
                                     drop_last=True)
    ds_test = torch.utils.data.DataLoader(testset,
                                          batch_size=1,
                                          num_workers=int(args['nW_eval']),
                                          shuffle=False,
                                          drop_last=False)
    results_dir = os.path.join('results', 'gpu' + args['folder'])
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'results.csv')
    best_path = os.path.join(results_dir, 'best.csv')
    box_path = os.path.join(results_dir, 'box.csv')
    f_all = open(results_path, 'w')
    f_best = open(best_path, 'w')
    f_box = open(box_path, 'w')
    f_all.write('epoches,label,acc\n')
    f_best.write('epoches,acc\n')
    f_box.write('epoches,label,acc1\n')

    path_acc1_best = os.path.join(results_dir, 'model_acc1_best.pth')
    path_acc_best = os.path.join(results_dir, 'model_acc_best.pth')
    path_last_epoch = os.path.join(results_dir, 'model_last_epoch.pth')
    best_acc1 = float('-inf')
    best_acc = float('-inf')
    best_sum = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    num_epochs = int(args['epoches'])
    for epoch in range(num_epochs):
        train(ds, model.train(), clip_model.eval(), optimizer_G, args)
        acc1, acc = inference_bbox(ds_test, model.eval(), clip_model.eval(), epoch, args)
        f_all.write(f"{epoch},test,{acc}\n")
        f_box.write(f"{epoch},test,{acc1}\n")
        f_all.flush();
        f_box.flush();
        f_best.flush()
        if acc1 > best_acc1:
            best_acc1 = acc1
            torch.save(model.state_dict(), path_acc1_best)
            print(f"[Epoch {epoch}] New best acc1 = {acc1:.4f}, model saved to: {path_acc1_best}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), path_acc_best)
            print(f"[Epoch {epoch}] New best acc = {acc:.4f}, model saved to: {path_acc_best}")
        torch.save(model.state_dict(), path_last_epoch)
        print(f" [Epoch {epoch}] Model saved as latest: {path_last_epoch}")
        current_sum = acc + acc1
        if current_sum > best_sum:
            best_sum = current_sum
            torch.save(model, args['path_best'])  # 注意：这里保存的是整个模型对象
            f_best.write(f"{epoch},{acc}\n")
            f_best.flush()
            print(
                f" [Epoch {epoch}] New best (acc+acc1) = {current_sum:.4f}, full model saved to: {args['path_best']}")
    f_all.close()
    f_box.close()
    f_best.close()
    print("\n" + "=" * 50)
    print(" Training completed.")
    print(f"   Final model saved at: {path_last_epoch}")
    print(f"   Best acc1 model saved at: {path_acc1_best} (acc1={best_acc1:.4f})")
    print(f"   Best acc model saved at: {path_acc_best} (acc={best_acc:.4f})")
    print("=" * 50)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-lr', '--learning_rate', default=0.0012, help='learning_rate', required=False)
    parser.add_argument('-bs', '--Batch_size', default=48, help='batch_size', required=False)
    parser.add_argument('-epoches', '--epoches', default=5000, help='number of epoches', required=False)
    parser.add_argument('-nW', '--nW', default=0, help='number of workers', required=False)
    parser.add_argument('-nW_eval', '--nW_eval', default=0, help='number of workers', required=False)
    parser.add_argument('-WD', '--WD', default=1e-4, help='weight decay', required=False)
    parser.add_argument('-order_ae', '--order_ae', default=16, help='order of the backbone - ae', required=False)
    parser.add_argument('-backbone', '--backbone', default='vgg', help='order of the backbone - ae', required=False)
    parser.add_argument('-task', '--task', default='vg_train', help='dataset task', required=False)
    parser.add_argument('-dataset', '--dataset', default='flicker', help='dataset task', required=False)
    parser.add_argument('-Isize', '--Isize', default=304, help='image size', required=False)
    parser.add_argument('-nC', '--nC', default=200, help='number of classes', required=False)
    parser.add_argument('-th', '--th', default=0.1, help='evaluation th', required=False)
    parser.add_argument('-temp', '--temp', default=1, help='pretrined models', required=False)
    parser.add_argument('-w0', '--w0', default=0.25, help='pretrined models', required=False)
    parser.add_argument('-w1', '--w1', default=4, help='pretrined models', required=False)
    parser.add_argument('-w2', '--w2', default=0.5, help='pretrined models', required=False)
    parser.add_argument('-w3', '--w3', default=0.25, help='pretrined models', required=False)
    parser.add_argument('-w4', '--w4', default=1, help='pretrined models', required=False)
    parser.add_argument('-w5', '--w5', default=0.01, help='pretrined models', required=False)
    parser.add_argument('-M', '--M', default=0.9, help='pretrined models', required=False)
    parser.add_argument('-prob', '--prob', default=10, help='pretrined models', required=False)
    parser.add_argument('-step_size', '--step_size', default=20, help='pretrined models', required=False)
    parser.add_argument('-gamma', '--gamma', default=1, help='pretrined models', required=False)
    parser.add_argument('-resume', '--resume', default=False, help='pretrined models', required=False)
    parser.add_argument('-resume_folder', '--resume_folder', default='488', help='pretrined models', required=False)
    parser.add_argument('-pretrained', '--pretrained', default=False, help='pretrined models', required=False)
    parser.add_argument('-img_path', '--img_path', default=True, help='pretrined models', required=False)
    parser.add_argument('-data_path', '--data_path',
                        default='/data/lpy1/EveCLIP/data/visual_genome', help='data set path', required=False)
    parser.add_argument('-val_path', '--val_path',
                        default=r'/data/lpy1/EveCLIP/data/flickr30k', help='data set path', required=False)
    args = vars(parser.parse_args())
    folder = open_folder('results')
    Isize = str(args['Isize'])
    args['folder'] = folder
    args['path_best'] = os.path.join('results', 'gpu' + folder, 'net_best.pth')
    args['resume_folder'] = os.path.join('results', 'gpu' + args['resume_folder'], 'net_best.pth')
    args['path_save_init'] = os.path.join('results', 'gpu' + folder, 'net_init.pth')
    args['path_init'] = os.path.join('results', 'init', 'cub',
                                     str(args['backbone']) + str(args['order_ae']), 'net_init.pth')
    main(args=args)

