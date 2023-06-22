from recbole.quick_start import load_data_and_model
import numpy as np
import pandas as pd
import os
import torch, gc

from util import load_setting



PATH = "/opt/ml/movie_rec_project/level2_movierecommendation-recsys-06/recbole_ex/settings.yaml"


def main():
    setting = load_setting(PATH)
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        setting["path"]["save_model"]
    )
    neg_sample_number = config["eval_neg_sample_args"]["sample_num"]
    
    # device 설정
    device = config.final_config_dict["device"]
    # user, item id -> token 변환 array
    user_id2token = dataset.field2id_token["user_id"]
    item_id2token = dataset.field2id_token["item_id"]

    # user-item sparse matrix
    matrix = dataset.inter_matrix(form="csr")

    # user id, predict item id 저장 변수
    pred_list = None
    user_list = None

    model.eval()
    
    i = 0
    temp = False
    for data in test_data:
        interaction = data[0].to(device)
        
        
        try:
            score = model.full_sort_predict(interaction)
        except NotImplementedError:
            batch_user = interaction["user_id"].cpu().numpy()
            # print(f"length of batch_user : {len(batch_user)}")
            batch_user = batch_user[0::neg_sample_number+1]
            # print(batch_user)
            input_inter = interaction
            len_input_inter = len(interaction)
            input_inter = input_inter.repeat(dataset.item_num)
            input_inter.update(dataset.get_item_feature().repeat(len_input_inter))  # join item feature
            input_inter = input_inter.to(config['device'])
            score = model.predict(input_inter)
            score = score.view(-1,len(item_id2token))
            temp = True
            
        # print(f"score = {score}")
        # print(f"score.shape = {score.shape}")
        
        
        
        
        i += 1
        if i % 1000 == 0:
            print(f'{i//1000} / {31}')
        # print("Shape 변환 후")
        # print(f"score = {score}")
        # print(f"score.shape = {score.shape}")
        
        # print(f"batch user = {interaction['user_id'].cpu().numpy()}")

        rating_pred = score.cpu().data.numpy().copy()
        batch_user_index = interaction["user_id"].cpu().numpy()
        rating_pred[matrix[batch_user_index].toarray() > 0] = 0
        if temp:
            # print(f'rating_pred = {rating_pred}\n rating_pred_shape : {rating_pred.shape}')
            rating_pred = rating_pred[0::neg_sample_number+1]
            # print("변환후")
            # print(f'rating_pred = {rating_pred}\n rating_pred_shape : {rating_pred.shape}')
            
        ind = np.argpartition(rating_pred, -10)[:, -10:]

        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]

        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]

        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
        # 예측값 저장
        if pred_list is None:
            pred_list = batch_pred_list
            user_list = batch_user
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)
            user_list = np.append(user_list, batch_user, axis=0)
        
        # print(pred_list)
        # print(user_list)
        # if i ==3:
        #     exit()        

    result = []
    for user, pred in zip(user_list, pred_list):
        for item in pred:
            result.append((int(user_id2token[user]), int(item_id2token[item])))

    # 데이터 저장
    dataframe = pd.DataFrame(result, columns=["user", "item"])
    dataframe.sort_values(by="user", inplace=True)
    dataframe.to_csv(
        os.path.join(setting["path"]["submission"], "submission_deepfm_dv6.csv"), index=False
    )
    print("inference done!")


if __name__ == "__main__":

    main()
