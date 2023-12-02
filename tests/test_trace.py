import torch
# assert head // ((head * 2) // batch_size) % 2 == 0
# print(head // other // batch_size)
# assert head // other // batch_size % 2 == 0


# assert other // head % 2 == 0
def test_batch_split_for_attention():
    other_head_dims = [10, 20, 40]
    head_dims = [5, 10, 20]
    batch_sizes = [2, 4, 6, 8, 10]

    for other, head in zip(other_head_dims, head_dims):
        for batch_size in range(2, 100, 2):
            # a = torch.rand(int(other * (batch_size / 2)), 1)
            b = torch.rand(int(head * (batch_size / 2)), 1)

            # print(b.size(0), batch_size)
            assert b.size(0) % (batch_size // 2) == 0
            # assert b.vsplit(b.size(0) // (batch_size // 2))

            # assert b.vsplit((a.size(0) // b.size(0)) % batch_size)


test_batch_split_for_attention()
