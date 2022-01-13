from typing import *
from abc import abstractmethod
import pandas as pd
from utils.data_utils import InputExample
import os

class DataProcessor:
    """
    labels of the dataset is optional 
    ``DataProcessor(data_path = 'datasets/')``
    labels file should have label names seperated by \n characters, such as
    ..  code-block:: 
        positive
        neutral
        negative
    Args:
        data_path (:obj:`str`, optional): Defaults to None. load labels from :obj:`data_path`. 
    """

    def __init__(self,
                 data_path: Optional[str] = None
                ):
        self.data_path = data_path
        labels = []
        labels_path = os.path.join(data_path, 'classes.txt')
        with open(labels_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    labels.append(line.strip())
        self.labels = labels

    @property
    def labels(self) -> List[Any]:
        if not hasattr(self, "_labels"):
            raise ValueError("DataProcessor doesn't set labels or label_mapping yet")
        return self._labels
        
    @labels.setter
    def labels(self, labels: Sequence[Any]):
        if labels is not None:
            self._labels = labels
            self._label_mapping = {k: i for (i, k) in enumerate(labels)}

    @property
    def label_mapping(self) -> Dict[Any, int]:
        if not hasattr(self, "_labels"):
            raise ValueError("DataProcessor doesn't set labels or label_mapping yet")
        return self._label_mapping

    @label_mapping.setter
    def label_mapping(self, label_mapping: Mapping[Any, int]):
        self._labels = [item[0] for item in sorted(label_mapping.items(), key=lambda item: item[1])]
        self._label_mapping = label_mapping
    
    @property
    def id2label(self) -> Dict[int, Any]:
        if not hasattr(self, "_labels"):
            raise ValueError("DataProcessor doesn't set labels or label_mapping yet")
        return {i: k for (i, k) in enumerate(self._labels)}


    def get_label_id(self, label: Any) -> int:
        """get label id of the corresponding label
        Args:
            label: label in dataset
        Returns:
            int: the index of label
        """
        return self.label_mapping[label] if label is not None else None

    def get_labels(self) -> List[Any]:
        """get labels of the dataset
        Returns:
            List[Any]: labels of the dataset
        """
        return self.labels
    
    def get_num_labels(self):
        """get the number of labels in the dataset
        Returns:
            int: number of labels in the dataset
        """
        return len(self.labels)

    def get_train_examples(self) -> InputExample:
        """
        get train examples from the training file under :obj:`data_dir`
        call ``get_examples(data_dir, "train")``, see :py:meth:`~openprompt.data_utils.data_processor.DataProcessor.get_examples`
        """
        return self.get_examples("train")

    def get_dev_examples(self) -> List[InputExample]:
        """
        get dev examples from the development file under :obj:`data_dir`
        call ``get_examples(data_dir, "dev")``, see :py:meth:`~openprompt.data_utils.data_processor.DataProcessor.get_examples`
        """
        return self.get_examples("dev")

    def get_test_examples(self) -> List[InputExample]:
        """
        get test examples from the test file under :obj:`data_dir`
        call ``get_examples(data_dir, "test")``, see :py:meth:`~openprompt.data_utils.data_processor.DataProcessor.get_examples`
        """
        return self.get_examples("test")

    def get_unlabeled_examples(self) -> List[InputExample]:
        """
        get unlabeled examples from the unlabeled file under :obj:`data_dir`
        call ``get_examples(data_dir, "unlabeled")``, see :py:meth:`~openprompt.data_utils.data_processor.DataProcessor.get_examples`
        """
        return self.get_examples("unlabeled")

    @abstractmethod
    def get_examples(self, split: Optional[str] = None) -> List[InputExample]:
        """get the :obj:`split` of dataset under :obj:`data_dir`
        :obj:`data_dir` is the base path of the dataset, for example:
        training file could be located in ``data_dir/train.txt``
        Args:
            data_dir (str): the base path of the dataset
            split (str): ``train`` / ``dev`` / ``test`` / ``unlabeled``
        Returns:
            List[InputExample]: return a list of :py:class:`~openprompt.data_utils.data_utils.InputExample`
        """
        raise NotImplementedError



class AgnewsProcessor(DataProcessor):
    def __init__(self, data_path):
        super().__init__(data_path=data_path)
        
    def get_examples(self, split=None):
        path = os.path.join(self.data_path, '{}.csv'.format(split))
        examples = []
        data = pd.read_csv(path, header=None)
        for i in range(len(data)):
            label = data.iloc[i][0] - 1
            
            if data.iloc[i][1] != data.iloc[i][1]:
                title = ''
            else:
                title = data.iloc[i][1]
                
            if data.iloc[i][2] != data.iloc[i][2]:
                content = ''
            else:
                content = data.iloc[i][2]
            
            example = InputExample(guid=str(i), text_a=title, text_b=content, label=label)
            examples.append(example)

        return examples
    

class YahooAnswersProcessor(DataProcessor):
    def __init__(self, data_path):
        super().__init__(data_path=data_path)
        
    def get_examples(self, split=None):
        path = os.path.join(self.data_path, '{}.csv'.format(split))
        examples = []
        data = pd.read_csv(path, header=None)
        for i in range(len(data)):
            if data.iloc[i][0] != data.iloc[i][0]:
                label = 0
            else:
                label = data.iloc[i][0] - 1
            
            if data.iloc[i][1] != data.iloc[i][1]:
                question_1 = ''
            else:
                question_1 = data.iloc[i][1]
                
            if data.iloc[i][2] != data.iloc[i][2]:
                question_2 = ''
            else:
                question_2 = data.iloc[i][2]
            
            if data.iloc[i][3] != data.iloc[i][3]:
                answer = ''
            else:
                answer = data.iloc[i][3]
            
            question = question_1 + ' ' + question_2
            
            example = InputExample(guid=str(i), text_a=question, text_b=answer, label=label)
            examples.append(example)

        return examples

class DBPediaProcessor(DataProcessor):
    def __init__(self, data_path):
        super().__init__(data_path=data_path)
        
    def get_examples(self, split=None):
        data_path = os.path.join(self.data_path, '{}.txt'.format(split))
        label_path = os.path.join(self.data_path, '{}_labels.txt'.format(split))
        examples = []
        with open(data_path, 'r') as f:
            data = f.readlines()
        with open(label_path, 'r') as f:
            labels = f.readlines()
        assert len(data) == len(labels)

        for i in range(len(data)):
            label = int(labels[i].strip())
            entity = data[i].split('.', 1)[0]
            if data[i].startswith('...'):
                entity = "..." + data[i][3:].split('.', 1)[0]
            if data[i].startswith('. . .'):
                entity = ". . . " + data[i][6:].split('.', 1)[0]
            
            example = InputExample(guid=str(i), text_a=data[i].strip(), text_b=entity, label=label)
            examples.append(example)

        return examples
