import json
from collections import defaultdict

import numpy as np
#from scipy.spatial.distance import cosine

train_ratio = 0.9 # ratio of train2004 allocated to training set, remainder becomes val


train2014_annot = json.load(open("mscoco_train2014_annotations.json"))
train2014_quest = json.load(open("OpenEnded_mscoco_train2014_questions.json"))
val2014_annot = json.load(open("mscoco_val2014_annotations.json"))
val2014_quest = json.load(open("OpenEnded_mscoco_val2014_questions.json"))


q_types = list(train2014_annot["question_types"].keys())

train2014_annot_by_qtype = [[] for _ in q_types]
for annot in train2014_annot["annotations"]:
    train2014_annot_by_qtype[q_types.index(annot["question_type"])].append(annot["question_id"])

seed = 2021
np.random.seed(seed)
train_ids = [id for qtype_ids in train2014_annot_by_qtype for id in np.random.choice(qtype_ids, int(len(qtype_ids)*train_ratio), replace=False)]
train_annot = [annot for annot in train2014_annot["annotations"] if annot["question_id"] in set(train_ids)]
val_annot = [annot for annot in train2014_annot["annotations"] if annot["question_id"] not in set(train_ids)]

# Debug check that train, val and test have the same question type distribution - train2014 and val2014 (i.e. test) do
# val_q_dist = {k: 0 for k in q_types}
# for annot in val2014_annot["annotations"]:
#     val_q_dist[annot["question_type"]] += 1
# train2014_q_dist = {k: 0 for k in q_types}
# for annot in train2014_annot["annotations"]:
#     train2014_q_dist[annot["question_type"]] += 1
# train_q_dist = {k: 0 for k in q_types}
# for annot in train_annot:
#     train_q_dist[annot["question_type"]] += 1
# val_q_dist = {k: 0 for k in q_types}
# for annot in val_annot:
#     val_q_dist[annot["question_type"]] += 1
# train2014_test_cos_dist = cosine(list(train2014_q_dist.values()), list(val_q_dist.values())) # close to 0
# train_test_cos_dist = cosine(list(train_q_dist.values()), list(val_q_dist.values())) # close to 0
# val_test_cos_dist = cosine(list(val_q_dist.values()), list(val_q_dist.values())) # close to 0

id2q = {q["question_id"]: q["question"] for q in train2014_quest["questions"]}
id2q.update({q["question_id"]: q["question"] for q in val2014_quest["questions"]})


def to_split_entry(annot, dsplit):
    label = defaultdict(list)
    # for gtAnsDatum in gts[quesId]['answers']:
    #   otherGTAns = [item for item in gts[quesId]['answers'] if item != gtAnsDatum]
    #   matchingAns = [item for item in otherGTAns if item['answer'] == resAns]
    #   acc = min(1, float(len(matchingAns)) / 3)
    #   gtAcc.append(acc)
    gtAnswers = [ans['answer'] for ans in annot["answers"]]
    for ans in annot["answers"]:
        otherGTAns = [item for item in annot["answers"] if item != ans]
        for a in gtAnswers:
            matchingAns = [item for item in otherGTAns if item['answer'] == a]
            acc = min(1, float(len(matchingAns)) / 3)
            label[a].append(acc)
    label = {k: round(np.mean(v), 1) for k,v in label.items()} # round to change all the 0.59999999999 to 0.6 for smaller file

    return {
        "answer_type": annot["answer_type"],
        "question_type": annot["question_type"],
        "answers": annot["answers"],
        "question_id": annot["question_id"],
        "img_id": "COCO_{}_{:012d}".format(dsplit, annot["image_id"]),
        "sent": id2q[annot["question_id"]],
        "label": label,
        "is_topk_optimal": True
    }

def no_empty_ans(annot):
    # 15 questions in total removed
    for ans in annot["answers"]:
        if not ans["answer"]:
            print("Empty answer")
            print(annot)
            return False
    return True

train = [to_split_entry(annot, "train2014") for annot in train_annot if no_empty_ans(annot)]
val = [to_split_entry(annot, "train2014") for annot in val_annot if no_empty_ans(annot)]
test = [to_split_entry(annot, "val2014") for annot in val2014_annot["annotations"] if no_empty_ans(annot)]

json.dump(train, open("train_{}-{}.json".format(int(train_ratio*10), seed), "w", encoding="utf-8"))
json.dump(val, open("val_{}-{}.json".format(int(train_ratio*10), seed), "w", encoding="utf-8"))
json.dump(test, open("test.json", "w", encoding="utf-8"))
