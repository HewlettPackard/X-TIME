// https://github.com/catboost/tutorials/blob/master/model_analysis/model_export_as_json_tutorial.ipynb
use rayon::prelude::*;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Model {
    features_info: FeaturesInfo,
    model_info: ModelInfo,
    oblivious_trees: Vec<Tree>,
}

impl Model {
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

#[derive(Debug, Deserialize)]
struct FeaturesInfo {
    float_features: Vec<FloatFeature>,
}

#[derive(Debug, Deserialize)]
struct FloatFeature {}

#[derive(Debug, Deserialize)]
struct ModelInfo {
    class_params: ClassParams,
}

#[derive(Debug, Deserialize)]
struct ClassParams {
    class_names: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct Tree {
    leaf_values: Vec<f64>,
    splits: Vec<TreeSplit>,
}

#[derive(Debug, Deserialize)]
struct TreeSplit {
    border: f64,
    float_feature_index: usize,
}

pub fn compile_model(model: &Model) -> (Vec<f64>, (usize, usize)) {
    let num_feature = model.features_info.float_features.len();
    let num_class = match model.model_info.class_params.class_names.len() {
        // representing binary classification with a single class as it also has a single value per leaf
        x if x < 3 => 1,
        x => x,
    };

    let trees = model
        .oblivious_trees
        .par_iter()
        .enumerate()
        .map(|(tree_id, tree)| compile_tree(tree, tree_id, num_feature, num_class))
        .collect::<Vec<_>>();

    let cam: Vec<f64> = trees
        .into_iter()
        .flat_map(|tree| tree.into_iter())
        .collect();

    let cam_width = num_feature * 2 + num_class + 2;
    let cam_height = cam.len() / cam_width;

    (cam, (cam_height, cam_width))
}

fn compile_tree(tree: &Tree, tree_id: usize, num_feature: usize, num_class: usize) -> Vec<f64> {
    let mut cam = Vec::new();
    if tree.leaf_values.is_empty() {
        return cam;
    }

    let cam_width = num_feature * 2;

    for leaf_index in 0..(tree.leaf_values.len() / num_class) {
        let mut row = vec![f64::NAN; cam_width + num_class + 2];

        let leaf_values_src =
            &tree.leaf_values[leaf_index * num_class..(leaf_index + 1) * num_class];
        let leaf_values_dest = &mut row[cam_width..cam_width + num_class];
        leaf_values_dest.copy_from_slice(leaf_values_src);

        row[cam_width + num_class] = 0.0;
        row[cam_width + num_class + 1] = tree_id as f64;

        for (depth, split) in tree.splits.iter().enumerate() {
            if (leaf_index >> depth) & 1 == 1 {
                let prev = row[split.float_feature_index * 2];
                if prev.is_nan() || split.border > prev {
                    row[split.float_feature_index * 2] = split.border;
                }
            } else {
                let prev = row[split.float_feature_index * 2 + 1];
                if prev.is_nan() || split.border < prev {
                    row[split.float_feature_index * 2 + 1] = split.border;
                }
            }
        }

        cam.append(&mut row);
    }

    cam
}
