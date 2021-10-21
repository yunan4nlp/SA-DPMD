from data.Dataloader import *

def evaluation(gold_file, predict_file):

    gold_instances = read_corpus(gold_file)
    predict_instances = read_corpus(predict_file)
    assert len(gold_instances) == len(predict_instances)
    uas_metric, las_metric = Metric(), Metric()
    for g_instance, p_instance in zip(gold_instances, predict_instances):
        g_arcs = get_arcs(g_instance)
        p_arcs = get_arcs(p_instance)
        uas_metric.correct_label_count += len(g_arcs & p_arcs)
        uas_metric.overall_label_count += len(g_arcs)
        uas_metric.predicated_label_count += len(p_arcs)

        g_rels = get_rels(g_instance)
        p_rels = get_rels(p_instance)

        las_metric.correct_label_count += len(g_rels & p_rels)
        las_metric.overall_label_count += len(g_rels)
        las_metric.predicated_label_count += len(p_rels)

        assert len(g_arcs) == len(g_rels)
    print("UAS:")
    uas_metric.print()
    print("LAS:")
    las_metric.print()
    return las_metric.getAccuracy()

def get_arcs(instance):
    arcs = set()
    for cur_y, relations in enumerate(instance.real_relations):
        if len(relations) == 0:
            y = str(cur_y)
            x = str(-1)
            arc = y + "##" + x
            arcs.add(arc)
        else:
            for relation in relations:
                y = str(relation['y'])
                x = str(relation['x'])
                arc = y + "##" + x
                arcs.add(arc)
    return arcs

def get_rels(instance):
    rels = set()
    for cur_y, relations in enumerate(instance.real_relations):
        if len(relations) == 0:
            y = str(cur_y)
            x = str(-1)
            rel = y + "##" + x + "##" + '<root>'
            rels.add(rel)
        else:
            for relation in relations:
                y = str(relation['y'])
                x = str(relation['x'])
                rel = y + "##" + x + "##" + relation['type']
                rels.add(rel)
    return rels




class Metric:
    def __init__(self):
        self.overall_label_count = 0
        self.correct_label_count = 0
        self.predicated_label_count = 0

    def reset(self):
        self.overall_label_count = 0
        self.correct_label_count = 0
        self.predicated_label_count = 0

    def bIdentical(self):
        if self.predicated_label_count == 0:
            if self.overall_label_count == self.correct_label_count:
                return True
            return False
        else:
            if self.overall_label_count == self.correct_label_count and \
                    self.predicated_label_count == self.correct_label_count:
                return True
            return False

    def getAccuracy(self):
        if self.overall_label_count + self.predicated_label_count == 0:
            return 1.0
        if self.predicated_label_count == 0:
            return self.correct_label_count*1.0 / self.overall_label_count
        else:
            return self.correct_label_count*2.0 / (self.overall_label_count + self.predicated_label_count)

    def print(self):
        if self.predicated_label_count == 0:
            print("Accuracy:\tP=" + str(self.correct_label_count) + '/' + str(self.overall_label_count))
        else:
            print("Recall:\tP=" + str(self.correct_label_count) + "/" + str(self.overall_label_count) + "=" + str(self.correct_label_count*1.0 / self.overall_label_count), end=",\t")
            print("Accuracy:\tP=" + str(self.correct_label_count) + "/" + str(self.predicated_label_count) + "=" + str(self.correct_label_count*1.0 / self.predicated_label_count), end=",\t")
            print("Fmeasure:\t" + str(self.correct_label_count*2.0 / (self.overall_label_count + self.predicated_label_count)))





