import os
import time
import json
import multiprocessing as mp
import torch
import math
import numpy as np
import random
from kgs import *
from model import *
from utils import *
from evaluation import *
from loss import *


def load_args(file):
    with open(file, 'r') as f:
        args_dict = json.load(f)
    args = ARGs(args_dict)
    return args

class ARGs:
    def __init__(self, dic):
        for k, v in dic.items():
            setattr(self, k, v)


def save_model(model, out_path):
    torch.save(model.state_dict(), out_path+'')


def test(epoch, model, kgs, args, save=True):
    model.eval()
    with torch.no_grad():
        e1 = torch.LongTensor(kgs.test_entities1).cuda()
        e2 = torch.LongTensor(kgs.test_entities2).cuda()
        
        at_mask1 = torch.ByteTensor([kgs.attr_mask[x] for x in kgs.test_entities1]).cuda()
        at_mask2 = torch.ByteTensor([kgs.attr_mask[x] for x in kgs.test_entities2]).cuda()
        im_mask1 = torch.ByteTensor([kgs.image_mask[x] for x in kgs.test_entities1]).cuda()
        im_mask2 = torch.ByteTensor([kgs.image_mask[x] for x in kgs.test_entities2]).cuda()

        embeds1, embeds2, e_r1, e_r2, e_i1, e_i2, e_a1, e_a2 = model.predict(e1, e2, at_mask1, im_mask1, at_mask2, im_mask2)

        rest_12, hits1_12, mr_12, all_mrr_12 = greedy_alignment(embeds1, embeds2, args.top_k, args.test_threads_num,
                                metric=args.eval_metric, normalize=args.eval_norm, csls_k=0, accurate=True)
                    
        greedy_alignment(embeds1, embeds2, args.top_k, args.test_threads_num, metric=args.eval_metric,
                        normalize=args.eval_norm, csls_k=args.csls, accurate=True)
        
    return all_mrr_12

def valid(epoch, model, kgs, args):
    model.eval()
    with torch.no_grad():
        e1 = torch.LongTensor(kgs.valid_entities1).cuda()
        e2 = torch.LongTensor(kgs.valid_entities2).cuda()

        at_mask1 = torch.ByteTensor([kgs.attr_mask[x] for x in kgs.valid_entities1]).cuda()
        at_mask2 = torch.ByteTensor([kgs.attr_mask[x] for x in kgs.valid_entities2]).cuda()
        im_mask1 = torch.ByteTensor([kgs.image_mask[x] for x in kgs.valid_entities1]).cuda()
        im_mask2 = torch.ByteTensor([kgs.image_mask[x] for x in kgs.valid_entities2]).cuda()

        embeds1, embeds2, e_r1, e_r2, e_i1, e_i2, e_a1, e_a2  = model.predict(e1, e2, at_mask1, im_mask1, at_mask2, im_mask2)

        _, hits1_12, mr_12, mrr_12 = greedy_alignment(embeds1, embeds2, args.top_k, args.test_threads_num,
                                                        args.eval_metric, args.eval_norm, csls_k=0, accurate=False)

    return hits1_12[0] if args.stop_metric == 'hits1' else mrr_12

def train(model, kgs, args, out_folder):
    t = time.time()
    relation_triples_num = len(kgs.relation_triples_list1) + len(kgs.relation_triples_list2)
    relation_triple_steps = int(math.ceil(relation_triples_num / args.batch_size))
    relation_step_tasks = task_divide(list(range(relation_triple_steps)), args.batch_threads_num)
    flag1, flag2 = -1, -1
    manager = mp.Manager()
    relation_batch_queue = manager.Queue()
    loss_list = []

    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr = args.learning_rate)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate)

    align_criterion = ContrastiveLoss()

    train_e1 = torch.LongTensor(kgs.train_entities1).cuda()
    train_e2 = torch.LongTensor(kgs.train_entities2).cuda()

    at_mask1 = torch.ByteTensor([kgs.attr_mask[x] for x in kgs.train_entities1]).cuda()
    at_mask2 = torch.ByteTensor([kgs.attr_mask[x] for x in kgs.train_entities2]).cuda()
    
    im_mask1 = torch.ByteTensor([kgs.image_mask[x] for x in kgs.train_entities1]).cuda()
    im_mask2 = torch.ByteTensor([kgs.image_mask[x] for x in kgs.train_entities2]).cuda()
    label_ = torch.eye(len(kgs.train_entities1)).cuda()


    # iterative
    rest_set_1 = [e1 for e1 in (kgs.valid_entities1 + kgs.test_entities1)]
    rest_set_2 = [e2 for e2 in (kgs.valid_entities2 + kgs.test_entities2)]
    np.random.shuffle(rest_set_1)
    np.random.shuffle(rest_set_2)

    
    max_mrr = 0
    new_e1 = []
    new_e2 = []
    for i in range(1, args.max_epoch + 1):
        print('epoch {}:'.format(i))
        start = time.time()
        epoch_loss = 0
        epoch_rloss = 0
        epoch_celoss = 0
        epoch_sup_celoss = 0
        epoch_orth_loss = 0
        epoch_mmd_loss = 0
        trained_samples_num = 0

        model.train()
        for steps_task in relation_step_tasks:
            mp.Process(target=generate_relation_triple_batch_queue,
                        args=(kgs.relation_triples_list1, kgs.relation_triples_list2,
                                kgs.relation_triples_set1, kgs.relation_triples_set2,
                                kgs.kg1_entities_list, kgs.kg2_entities_list,
                                args.batch_size, steps_task,
                                relation_batch_queue, args.neg_triple_num)).start()
        for _ in range(relation_triple_steps):
            optimizer.zero_grad()
            batch_pos, batch_neg = relation_batch_queue.get()
            rel_p_h = torch.LongTensor([x[0] for x in batch_pos]).cuda()
            rel_p_r = torch.LongTensor([x[1] for x in batch_pos]).cuda()
            rel_p_t = torch.LongTensor([x[2] for x in batch_pos]).cuda()
            rel_n_h = torch.LongTensor([x[0] for x in batch_neg]).cuda()
            rel_n_r = torch.LongTensor([x[1] for x in batch_neg]).cuda()
            rel_n_t = torch.LongTensor([x[2] for x in batch_neg]).cuda()
            r_loss = model(rel_p_h, rel_p_r, rel_p_t, rel_n_h, rel_n_r, rel_n_t)
             
            rs, ats, ims, score, orth_loss, mmd_loss = model.predict(train_e1, train_e2, at_mask1, im_mask1, at_mask2, im_mask2, 'train')
            align_loss = align_criterion(score, label_) + align_criterion(rs, label_) + align_criterion(ats, label_) + align_criterion(ims, label_)
            loss = r_loss + align_loss + orth_loss + mmd_loss   
            loss.backward()
            optimizer.step()
            trained_samples_num += len(batch_pos)

            epoch_rloss += r_loss.item()
            epoch_celoss += align_loss.item()
            epoch_orth_loss += orth_loss.item() 
            epoch_mmd_loss += mmd_loss.item() 


        if args.iterative and new_e1:
            sup_align_steps = int(math.ceil(len(new_e1) / args.e_batch_size))
            optimizer.zero_grad()
            sup_align_loss = 0
            for ind in range(sup_align_steps):
                start_ = ind*args.e_batch_size
                end_ = (ind+1)*args.e_batch_size
                print(start_, end_)
                
                train_e1 = torch.LongTensor(new_e1[start_:end_]).cuda()
                train_e2 = torch.LongTensor(new_e2[start_:end_]).cuda()

                at_mask1 = torch.ByteTensor([kgs.attr_mask[x] for x in new_e1[start_:end_]]).cuda()
                at_mask2 = torch.ByteTensor([kgs.attr_mask[x] for x in new_e2[start_:end_]]).cuda()
                im_mask1 = torch.ByteTensor([kgs.image_mask[x] for x in new_e1[start_:end_]]).cuda()
                im_mask2 = torch.ByteTensor([kgs.image_mask[x] for x in new_e2[start_:end_]]).cuda()

                label_ = torch.eye(len(new_e1[start_:end_])).cuda()
                 
                rs, ats, ims, score, _, _ = model.predict(train_e1, train_e2, at_mask1, im_mask1, at_mask2, im_mask2, 'train')
                sup_align_loss += align_criterion(score, label_) + align_criterion(rs, label_) + align_criterion(ats, label_) + align_criterion(ims, label_)
                
            sup_align_loss*=0.1
            sup_align_loss.backward()
            optimizer.step()
            epoch_sup_celoss += sup_align_loss.item()


        epoch_rloss /= trained_samples_num
        epoch_celoss /= len(train_e1)
        epoch_orth_loss /= len(train_e1)
        epoch_mmd_loss /= len(train_e1)

        if args.iterative and new_e1:
            epoch_sup_celoss /= len(new_e1)
        epoch_loss = epoch_rloss + epoch_celoss + epoch_sup_celoss + epoch_orth_loss + epoch_mmd_loss


        random.shuffle(kgs.relation_triples_list1)
        random.shuffle(kgs.relation_triples_list2)
        end = time.time()
        print('loss: {:.8f}, relation loss: {:.8f}, align loss:{:.8f}, orth loss: {:.8f}, mmd loss: {:.8f}, time: {:.4f}s'.format(epoch_loss,epoch_rloss, epoch_celoss, epoch_orth_loss, epoch_mmd_loss, end - start))

        loss_list.append(epoch_loss)


        if i >= args.start_valid and i > 0:
            flag = valid(i, model, kgs, args)
            test(i, model, kgs, args)
            if flag > max_mrr:
                torch.save(model.state_dict(), out_folder+'model_best.pkl')
                max_mrr = flag
                print('best {}'.format(i))
            flag1, flag2, stop = early_stop(flag1, flag2, flag)
            if args.early_stop and (stop or i == args.max_epoch):
                break
        
        if args.iterative and i >= args.start_valid and i % args.eval_freq == 0:
            new_pair = []
            Lvec, Rvec = get_embedding(model, kgs, rest_set_1, rest_set_2)
            Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
            Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
            A, _ = eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], 16, 0, True, False)
            B, _ = eval_alignment_by_sim_mat(Rvec, Lvec, [1, 5, 10], 16, 0, True, False)
            A = sorted(list(A))
            B = sorted(list(B))
            for a, b in A:
                if B[b][1] == a:
                    new_pair.append([rest_set_1[a], rest_set_2[b]])


            print("generate new semi-pairs: %d." % len(new_pair))
            new_e1 += [x[0] for x in new_pair]
            new_e2 += [x[1] for x in new_pair]

            sup_triples1_set, sup_triples2_set = kgs.generate_sup_relation_triples(new_pair,
                                                                kgs.rt_dict1, kgs.hr_dict1,
                                                                kgs.rt_dict2, kgs.hr_dict2)
            kgs.relation_triples_list1, kgs.relation_triples_list2 = kgs.add_sup_relation_triples(sup_triples1_set, sup_triples2_set)
            kgs.relation_triples_set1 = set(kgs.relation_triples_list1)
            kgs.relation_triples_set2 = set(kgs.relation_triples_list2)
            
            for e1, e2 in new_pair:
                if e1 in rest_set_1:
                    rest_set_1.remove(e1)
                if e2 in rest_set_2:
                    rest_set_2.remove(e2)

    print("Training ends. Total time = {:.3f} s.".format(time.time() - t))

def get_embedding(model, kgs, entity1, entity2):
    model.eval()
    with torch.no_grad():
        e1 = torch.LongTensor(entity1).cuda()
        e2 = torch.LongTensor(entity2).cuda()
        at_mask1 = torch.ByteTensor([kgs.attr_mask[x] for x in entity1]).cuda()
        at_mask2 = torch.ByteTensor([kgs.attr_mask[x] for x in entity2]).cuda()
        im_mask1 = torch.ByteTensor([kgs.image_mask[x] for x in entity1]).cuda()
        im_mask2 = torch.ByteTensor([kgs.image_mask[x] for x in entity2]).cuda()
        embeds1, embeds2, e_r1, e_r2, e_i1, e_i2, e_a1, e_a2 = model.predict(e1, e2, at_mask1, im_mask1, at_mask2, im_mask2)
    return np.array(embeds1), np.array(embeds2)

def generate_out_folder(out_folder, training_data_path, div_path, method_name):
    params = training_data_path.strip('/').split('/')
    path = params[-1]
    folder = out_folder + method_name + '/' + path + "/" + div_path + str(time.strftime("%Y%m%d%H%M%S")) + "/"
    print("results output folder:", folder)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    t = time.time()
    args = load_args('./args/tmea.json')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    out_folder = generate_out_folder(args.output, args.training_data, args.dataset_division, 'TMEA')

    kgs = KGs(args.training_data, args.dataset_division, ordered=True, modality='ria')
    model = TMEA(kgs, args)
    model.cuda()
    model = torch.compile(model)
    train(model, kgs, args, out_folder)
    model.load_state_dict(torch.load(out_folder + 'model_best.pkl'))
    mrr = test(args.max_epoch+1, model, kgs, args)

    print("Total run time = {:.3f} s.".format(time.time() - t))
