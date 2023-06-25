import unittest
import lm_dataset
import logging
import transformers

logger = logging.getLogger(__name__)

class Args:
    def __init__(self, data_path, max_seq_length, epoch_length ):
        self.data_path = data_path
        self.max_seq_length = max_seq_length
        self.epoch_length = epoch_length
    def __str__(self):
        return f'data_path:{self.data_path} max_seq_length:{self.max_seq_length} epoch_length:{self.epoch_length}'

class LMDataSetTestCase(unittest.TestCase):
    def setUp(self) -> None:
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler())

    def test_lm_dataset(self):
        args = Args("data/k8s/test.bin", 100, 10 )
        logger.info('arg is {}'.format(args))
        model_name = "THUDM/chatglm-6b"
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)

        ds = lm_dataset.LMDataset(args)
        for i in range(len(ds)):
            x, y = ds[i]
            logger.info('ds {} is {} {}'.format(i, x, y))
            data = tokenizer.decode(x)
            logger.info('ds {} is {}'.format(i, data))
        self.assertEqual(True, len(ds)>0)  # add assertion here

if __name__ == '__main__':
    unittest.main()
