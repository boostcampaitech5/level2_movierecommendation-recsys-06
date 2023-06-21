from recbole.quick_start import load_data_and_model
import numpy as np
import pandas as pd
import os

from util import load_setting


PATH = "/opt/ml/movie_rec_project/level2_movierecommendation-recsys-06/recbole_ex/settings.yaml"


def main():
    setting = load_setting(PATH)
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        setting["path"]["save_model"]
    )
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
    for data in test_data:
        interaction = data[0].to(device)
        score = model.full_sort_predict(interaction)

        rating_pred = score.cpu().data.numpy().copy()
        batch_user_index = interaction["user_id"].cpu().numpy()
        rating_pred[matrix[batch_user_index].toarray() > 0] = 0
        ind = np.argpartition(rating_pred, -10)[:, -10:]

        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]

        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]

        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
        # 예측값 저장
        if pred_list is None:
            pred_list = batch_pred_list
            user_list = batch_user_index
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)
            user_list = np.append(user_list, batch_user_index, axis=0)

    result = []
    for user, pred in zip(user_list, pred_list):
        for item in pred:
            result.append((int(user_id2token[user]), int(item_id2token[item])))

    # 데이터 저장
    dataframe = pd.DataFrame(result, columns=["user", "item"])
    dataframe.sort_values(by="user", inplace=True)
    dataframe.to_csv(
        os.path.join(setting["path"]["submission"], "submission.csv"), index=False
    )
    print("inference done!")


if __name__ == "__main__":
    main()
