from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    Creates dataset from input_data

    Index is the user number. All items are returned with the interaction data of the user.
    If a user didn't interact with the item then the item's value for the interaction will be None.
    """

    def __init__(self, input_interaction, input_user, input_item):
        self.input_interaction = input_interaction
        self.input_user = input_user
        self.input_item = input_item

    def __getitem__(self, index):
        # Get user ID from index and the user interaction data
        user_id = self.input_user["user_id"][index]
        user_df = self.input_user[self.input_user["user_id"] == user_id]

        # Merge all item data to the interaction data
        # The items without interaction data will be filled with None or pd.nan
        output_df = self.input_item.merge(
            self.input_interaction[self.input_interaction["user_id"] == user_id],
            how="left",
            on="item_id",
        )

        # Get dictionary of all the values
        user_data = {
            col: val for val, col in zip(user_df.values.squeeze(), user_df.columns)
        }
        output_data = [
            {col: val for val, col in zip(item, output_df.columns)}
            for item in output_df.values
        ]

        return user_data, output_data

    def __len__(self):
        # The length of the index can't be equal to or greater than the number of users
        return len(self.input_user)
