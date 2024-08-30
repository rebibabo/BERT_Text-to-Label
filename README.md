## This is an example program -- fine-tuning the Bert on text-to-label task

There are two main class, MyDataset(my_dataset.py) and Trainer(trainer.py), the main training function is run.py

### myDataset
The source file must be in 'jsonl', 'csv', or 'tsv' format, each of which must contained a attribution called 'label', for example, a possible jsonl file may be:
```
{"filename": "c_24_2221.c", "label": "24", "code": "void main() {\n    char s[10000];\n    int i, j, m = 0, n = 0, l;\n    gets(s);\n    l = strlen(s);\n    for (i = 0; i <= l - 1; i++) {\n        if (((s[i] >= 'a') && (s[i] <= 'z')) ||\n            ((s[i] >= 'A') && (s[i] <= 'Z'))) {\n            for (j = i; j <= l - 1; j++) {\n                if (((s[j] >= 'a') && (s[j] <= 'z')) ||\n                    ((s[j] >= 'A') && (s[j] <= 'Z'))) {\n                    if (j != l - 1)\n                        continue;\n                    else {\n                        if ((j - i + 1) > (m - n)) {\n                            m = j + 1;\n                            n = i;\n                        }\n                        i = j;\n                        break;\n                    }\n                } else {\n                    if ((j - i) > (m - n)) {\n                        m = j;\n                        n = i;\n                    }\n                    i = j - 1;\n                    break;\n                }\n            }\n        }\n    }\n    for (i = n; i <= m - 1; i++)\n        printf(\"%c\", s[i]);\n    printf(\"\\n\");\n    m = 100;\n    n = 0;\n    for (i = 0; i <= l - 1; i++) {\n        if (((s[i] >= 'a') && (s[i] <= 'z')) ||\n            ((s[i] >= 'A') && (s[i] <= 'Z'))) {\n            for (j = i; j <= l - 1; j++) {\n                if (((s[j] >= 'a') && (s[j] <= 'z')) ||\n                    ((s[j] >= 'A') && (s[j] <= 'Z'))) {\n                    if (j != l - 1)\n                        continue;\n                    else {\n                        if ((j - i + 1) < (m - n)) {\n                            m = j + 1;\n                            n = i;\n                        }\n                        i = j;\n                        break;\n                    }\n                } else {\n                    if ((j - i) < (m - n)) {\n                        m = j;\n                        n = i;\n                    }\n                    i = j - 1;\n                    break;\n                }\n            }\n        }\n    }\n    for (i = n; i <= m - 1; i++)\n        printf(\"%c\", s[i]);\n    printf(\"\\n\");\n}\n"}
```
The loading code is as followed, if the parameter 'seed' is set to True, then the random shuffle will be duplicatable, if the parameter 'save' is set to True, then the tokenizing result will be saved to the same source path with the name starting with "cache", so there is no more worries about tokenizing over and over again. 
```
from my_dataset import MyDataset
dataset = MyDataset('a.jsonl', seed=42, save=False)
```

Given that the csv or tsv file may not contain a header, like the example below
```
gj04	0	*	They drank the pub.
```
Your should give a parameter "name" to give each column a name
```
dataset = MyDataset('b.csv', delimiter='\t', header=None, names=['source', 'label', 'tag', 'sentence'])
```
