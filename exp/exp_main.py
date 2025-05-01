import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import FEDformer, Autoformer, Informer, Transformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import pandas as pd
from utils.timefeatures import time_features


warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'FEDformer': FEDformer,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        #70% train, 10% val, 20% test
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        dates = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)

                r_begin = i + self.args.seq_len - self.args.label_len
                r_end = r_begin + self.args.pred_len
                dates.append(test_data.date_list[r_begin:r_end].values)

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        # flatten to 2D: (B*pred_len, C)
        B, L, C = preds.shape
        preds_2d = preds.reshape(-1, C)
        trues_2d = trues.reshape(-1, C)

        # inverse‐scale both
        preds_inv_2d = test_data.inverse_transform_custom(preds_2d)
        trues_inv_2d = test_data.inverse_transform_custom(trues_2d)

        # reshape back to (B, L, C)
        preds_inv = preds_inv_2d.reshape(B, L, C)
        trues_inv = trues_inv_2d.reshape(B, L, C)

        # now compute metrics on the real‐scale values
        mae, mse, rmse, mape, mspe = metric(preds_inv, trues_inv)
        print('Inverse‐scaled mse:{}, mae:{}'.format(mse, mae))

        with open("result.txt", 'a') as f:
            f.write(setting + "\n")
            f.write(f"Inverse-scaled mse:{mse}, mae:{mae}\n\n")

        # optionally save the inverse‐scaled outputs too
        np.save(folder_path + 'pred_inv.npy', preds_inv)
        np.save(folder_path + 'true_inv.npy', trues_inv)
        

    # def predict(self, setting, load=False):
    #     pred_data, pred_loader = self._get_data(flag='pred')
    #     pred_len = pred_data.pred_len   # 8784 in your 1-year case
    #     label_len = self.args.label_len

    #     if load:
    #         path = os.path.join(self.args.checkpoints, setting)
    #         best_model_path = path + '/' + 'checkpoint.pth'
    #         self.model.load_state_dict(torch.load(best_model_path))

    #     preds = []

    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float()
    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)

    #             # # decoder input
    #             # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
    #             # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

    #             # build dec_inp using dataset.pred_len, not args.pred_len
    #             B, _, C     = batch_y.shape
    #             batch_y     = batch_y.float().to(self.device)
    #             dec_zeros   = torch.zeros((B, pred_len, C), device=self.device)
    #             warmup      = batch_y[:, :label_len, :]
    #             dec_inp     = torch.cat([warmup, dec_zeros], dim=1)

    #             # encoder - decoder
    #             # if self.args.use_amp:
    #             #     with torch.cuda.amp.autocast():
    #             #         if self.args.output_attention:
    #             #             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    #             #         else:
    #             #             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #             # else:
    #             #     if self.args.output_attention:
    #             #         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    #             #     else:
    #             #         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    #             if self.args.use_amp:
    #                 with torch.cuda.amp.autocast():
    #                     outputs = (self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    #                             if self.args.output_attention
    #                             else self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark))
    #             else:
    #                 outputs = (self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    #                         if self.args.output_attention
    #                         else self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark))
                    
    #             pred = outputs.detach().cpu().numpy()  # .squeeze()
    #             preds.append(pred)

    #     preds = np.array(preds)
    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

    #     # result save
    #     folder_path = './results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     np.save(folder_path + 'real_prediction.npy', preds)

    #     # inverse transform
    #     preds_inv_2d = pred_data.inverse_transform(preds)  # pakai scaler dari Dataset_Pred
    #     # reshape kembali
    #     B = len(pred_loader)       # atau simpan len di atas
    #     L = self.args.pred_len
    #     C = preds_inv_2d.shape[1]
    #     preds_inv = preds_inv_2d.reshape(B, L, C)

    #     np.save(folder_path + 'pred_inverse.npy',    preds_inv)

    #     pred_vals = preds_inv[0]
    #     col_names = ([self.args.target] 
    #                 if C==1 
    #                 else [f"{self.args.target}_{i}" for i in range(C)])
    #     df_pred = pd.DataFrame(pred_vals, columns=col_names)
    #     df_pred['date'] = pred_data.pred_dates
    #     df_pred.set_index('date', inplace=True)

    #     csv_path = os.path.join(folder_path, 'prediction_inverse.csv')
    #     df_pred.to_csv(csv_path)
    #     print(f"Saved inverse‐scaled prediction to {csv_path}")

    #     return

    def predict(self, setting, load=False):
        # 1) Prepare dataset and model checkpoint
        pred_data, _ = self._get_data(flag='pred')
        seq_len      = pred_data.seq_len
        label_len    = self.args.label_len
        pred_chunk   = self.args.pred_len  # chunk size per iteration
        total_steps  = len(pred_data.pred_dates)

        # Optionally load the trained model
        if load:
            ckpt = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(ckpt))
            print(f"Loaded model from {ckpt}")

        self.model.eval()
        results = []  # to collect each chunk of predictions

        # Initialize sliding window
        hist_x      = pred_data.data_x[-seq_len:]           # shape (seq_len, C)
        hist_stamp  = pred_data.data_stamp[-seq_len:]       # shape (seq_len, time_features)

        with torch.no_grad():
            for start in range(0, total_steps, pred_chunk):
                # Determine how many steps in this chunk
                cur_len = min(pred_chunk, total_steps - start)

                # Slice out input sequences
                input_x  = hist_x[-seq_len:]  # last seq_len points
                input_x  = torch.from_numpy(input_x).unsqueeze(0).float().to(self.device)

                # Build decoder input: [warmup_labels | zeros]
                warmup = input_x[:, -label_len:, :]
                zeros  = torch.zeros((1, cur_len, input_x.size(2)), device=self.device)
                dec_inp = torch.cat([warmup, zeros], dim=1)

                # Build encoder timestamp features
                enc_stamp = torch.from_numpy(hist_stamp).unsqueeze(0).float().to(self.device)

                # Build decoder timestamp features: last label_len + future cur_len
                hist_dec_stamp = torch.from_numpy(hist_stamp[-label_len:]).unsqueeze(0).float().to(self.device)
                future_dates = pred_data.pred_dates[start:start+cur_len]
                future_stamp = time_features(pd.DatetimeIndex(future_dates), freq=self.args.freq).T
                future_stamp = torch.from_numpy(future_stamp).unsqueeze(0).float().to(self.device)
                dec_stamp = torch.cat([hist_dec_stamp, future_stamp], dim=1)

                # Forward pass
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        out = (self.model(input_x, enc_stamp, dec_inp, dec_stamp)[0]
                               if self.args.output_attention
                               else self.model(input_x, enc_stamp, dec_inp, dec_stamp))
                else:
                    out = (self.model(input_x, enc_stamp, dec_inp, dec_stamp)[0]
                           if self.args.output_attention
                           else self.model(input_x, enc_stamp, dec_inp, dec_stamp))

                # Extract only the forecast portion
                pred_chunk_vals = out.cpu().numpy()[0, -cur_len:, :]
                results.append(pred_chunk_vals)

                # Slide window: drop old, append new
                hist_x     = np.concatenate([hist_x[cur_len:], pred_chunk_vals], axis=0)
                hist_stamp = np.concatenate([
                    hist_stamp[cur_len:],
                    future_stamp.cpu().numpy().squeeze(0)
                ], axis=0)

        # Concatenate all chunks: shape (total_steps, C)
        preds = np.concatenate(results, axis=0)

        # Inverse scale back to original units
        flat_preds = preds.reshape(-1, preds.shape[-1])
        if hasattr(pred_data, 'scaler') and pred_data.scaler is not None:
            inv_flat = pred_data.inverse_transform(flat_preds)
        else:
            inv_flat = flat_preds
        preds_inv = inv_flat.reshape(preds.shape)

        # Save results
        folder = f"./results/{setting}/"
        os.makedirs(folder, exist_ok=True)
        np.save(folder + 'pred_inverse.npy', preds_inv)

        # Export CSV
        col_names = pred_data.pred_cols  # e.g. ['Temp','Humidity','WindSpeed']
        df = pd.DataFrame(preds_inv, columns=col_names)
        df['date'] = pred_data.pred_dates
        df.to_csv(folder+'prediction_inverse.csv', index=False)
        print(f"Saved forecast CSV to {folder}prediction_inverse.csv")

        return preds, preds_inv
