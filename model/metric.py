import numpy as np


def get_metric_score(
    answer: np.ndarray, pred_list: np.ndarray, topks: list = [5, 10]
) -> tuple:
    """Get Recall@K & NDCG@K

    Args:
        answer (np.ndarray): Movies that were actually seen by batch users
        pred_list (np.ndarray): movies recommended to users on a batch
        topk (list, optional): Array of the number of movies to recommend  defaults to [5, 10].

    Returns:
        tuple: Recall list and NDCG list for k in topks
    """
    recall, ndcg = [], []
    for k in topks:
        recall.append(recall_at_k(answer, pred_list, k))
        ndcg.append(ndcg_at_k(answer, pred_list, k))
    return recall, ndcg


def recall_at_k(actual: np.ndarray, predicted: np.ndarray, topk: int) -> float:
    """Calculate Recall@K

    Args:
        actual (np.ndarray): Movies that were actually seen by batch users
        predicted (np.ndarray): movies recommended to users on a batch
        topk (int): The number of movies to recommend

    Returns:
        float: Average Recall@K in batch units
    """
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        k = min(len(act_set), topk)
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(k)
            true_users += 1
    return sum_recall / true_users


def ndcg_at_k(actual: np.ndarray, predicted: np.ndarray, topk: int) -> float:
    """Calculate NDCG@K

    Args:
        actual (np.ndarray): Movies that were actually seen by batch users
        predicted (np.ndarray): movies recommended to users on a batch
        topk (int): The number of movies to recommend

    Returns:
        float: Average NDCG@K in batch units
    """
    tp = 1.0 / np.log2(np.arange(2, topk + 2))
    result = 0
    number_of_batch_user = len(actual)
    for uid in range(number_of_batch_user):
        k = min(topk, len(actual[uid]))

        idcg = tp[:k].sum()
        dcg = sum(
            [
                int(predicted[user_id][j] in set(actual[user_id])) / math.log(j + 2, 2)
                for j in range(topk)
            ]
        )
        result += idcg / dcg
    return result / float(number_of_batch_user)


def precision_at_k(actual: np.ndarray, predicted: np.ndarray, topk: int) -> float:
    """Calculate Precision@K

    Args:
        actual (np.ndarray): Movies that were actually seen by batch users
        predicted (np.ndarray): movies recommended to users on a batch
        topk (int): The number of movies to recommend

    Returns:
        float: float: Average Precision@K in batch units
    """
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users
