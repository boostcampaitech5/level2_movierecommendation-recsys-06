from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, raw_input, raw_user, raw_item):
        self.raw_input = raw_input
        self.raw_user = raw_user
        self.raw_item = raw_item

    def __getitem__(self, index):
        user_id = self.raw_user['user_id'][index]
        user_df = self.raw_user[self.raw_user['user_id'] == user_id]

        output_df = self.raw_item.merge(self.raw_input[self.raw_input['user_id'] == user_id], how='left', on='item_id')

        user_data = {col: val for val, col in zip(user_df.values.squeeze(), user_df.columns)}
        output_data = [{col: val for val, col in zip(item, output_df.columns)} for item in output_df.values]

        return user_data, output_data

    def __len__(self):
        return len(self.raw_user)