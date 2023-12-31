from . import evaluators, extractors, matchers, ransacs, readers


def load_component(compo_name, model_name, config):
    if compo_name == "extractor":
        component = load_extractor(model_name, config)
    elif compo_name == "reader":
        component = load_reader(model_name, config)
    elif compo_name == "matcher":
        component = load_matcher(model_name, config)
    elif compo_name == "evaluator":
        component = load_evaluator(model_name, config)
    elif compo_name == "ransac":
        component = load_ransac(model_name, config)
    else:
        raise NotImplementedError
    return component


def load_extractor(model_name, config):
    if model_name == "root":
        extractor = extractors.ExtractSIFT(config)
    elif model_name == "sp":
        extractor = extractors.ExtractSuperpoint(config)
    elif model_name == "d2":
        extractor = extractors.ExtractD2Net(config)
    elif model_name == "loftr":
        extractor = extractors.ExtractLoFTR(config)
    elif model_name == "disk":
        extractor = extractors.ExtractDISK(config)
    elif model_name == "keyaff":
        extractor = extractors.ExtractKeyAff(config)
    else:
        raise NotImplementedError
    return extractor


def load_matcher(model_name, config):
    if model_name == "SGM":
        matcher = matchers.GNN_Matcher(config, "SGM")
    elif model_name == "SG":
        matcher = matchers.GNN_Matcher(config, "SG")
    elif model_name == "NN":
        matcher = matchers.NN_Matcher(config)
    elif model_name == "FLANN":
        matcher = matchers.FLANN_Matcher(config)
    elif model_name == "BF":
        matcher = matchers.BF_Matcher(config)
    elif model_name == "LightGlue":
        matcher = matchers.LightGlue_Matcher(config)
    elif model_name == "adalam":
        matcher = matchers.AdaLAM_Matcher(config)
    elif model_name == "LoFTR":
        matcher = None
    else:
        raise NotImplementedError
    return matcher


def load_reader(model_name, config):
    if model_name == "standard":
        reader = readers.standard_reader(config)
    else:
        raise NotImplementedError
    return reader


def load_evaluator(model_name, config):
    if model_name == "AUC":
        evaluator = evaluators.auc_eval(config)
    elif model_name == "FM":
        evaluator = evaluators.FMbench_eval(config)
    else:
        raise NotImplementedError
    return evaluator


def load_ransac(model_name, config):
    if model_name == "ransac":
        ransac = ransacs.RANSAC(config)
    elif model_name == "degensac":
        ransac = ransacs.Degensac(config)
    elif model_name == "magsacpp":
        ransac = ransacs.Magsacpp(config)
    else:
        raise NotImplementedError
    return ransac
