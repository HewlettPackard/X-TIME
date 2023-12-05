// https://github.com/dmlc/treelite/blob/v4/src/json_serializer.cc
use rayon::prelude::*;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Model {
    num_feature: usize,
    task_param: TaskParam,
    trees: Vec<Tree>,
}

impl Model {
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

#[derive(Debug, Deserialize)]
struct TaskParam {
    num_class: usize,
}

#[derive(Debug, Deserialize)]
struct Tree {
    nodes: Vec<Node>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum LeafVector {
    Scalar(f64),
    OneHot(Vec<f64>),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Node {
    Internal {
        split_feature_id: usize,
        comparison_op: ComparisonOp,
        threshold: f64,
        left_child: usize,
        right_child: usize,
    },
    Leaf {
        leaf_value: LeafVector,
    },
}

// see https://github.com/dmlc/treelite/blob/v4/include/treelite/base.h
#[derive(Debug, Deserialize, Clone, Copy)]
enum ComparisonOp {
    #[serde(rename = "<")]
    Less,
    #[serde(rename = ">")]
    Greater,
    #[serde(rename = "<=")]
    LessEquals,
    #[serde(rename = ">=")]
    GreaterEquals,
}

impl ComparisonOp {
    fn invert(self) -> Self {
        match self {
            Self::Less => Self::GreaterEquals,
            Self::Greater => Self::LessEquals,
            Self::LessEquals => Self::Greater,
            Self::GreaterEquals => Self::Less,
        }
    }
}

pub fn compile_model(model: &Model) -> (Vec<f64>, (usize, usize)) {
    let num_class = model.task_param.num_class;

    let trees = model
        .trees
        .par_iter()
        .enumerate()
        .map(|(tree_id, tree)| compile_tree(tree, tree_id, model.num_feature, num_class))
        .collect::<Vec<_>>();

    let cam: Vec<f64> = trees
        .into_iter()
        .flat_map(|tree| tree.into_iter())
        .collect();

    let cam_width = model.num_feature * 2 + 3;
    let cam_height = cam.len() / cam_width;

    (cam, (cam_height, cam_width))
}

fn compile_tree(tree: &Tree, tree_id: usize, num_feature: usize, num_class: usize) -> Vec<f64> {
    let mut cam = Vec::new();
    if tree.nodes.is_empty() {
        return cam;
    }

    let mut stack: Vec<(usize, Vec<usize>)> = vec![(0, vec![0])];

    while let Some((index, path)) = stack.pop() {
        match &tree.nodes[index] {
            Node::Internal {
                left_child,
                right_child,
                ..
            } => {
                let mut right_path = path.clone();
                right_path.push(*right_child);
                stack.push((*right_child, right_path));

                let mut left_path = path.clone();
                left_path.push(*left_child);
                stack.push((*left_child, left_path));
            }

            Node::Leaf { leaf_value } => {
                let cam_width = num_feature * 2;
                let mut row = vec![f64::NAN; cam_width + 3];

                match leaf_value {
                    LeafVector::Scalar(value) => {
                        row[cam_width] = *value;
                        row[cam_width + 1] = (tree_id % num_class) as f64;
                    }
                    LeafVector::OneHot(one_hot_vec) => {
                        let hot_idx = one_hot_vec
                            .iter()
                            .position(|value| *value != 0.0)
                            .expect("no hot value");
                        row[cam_width] = one_hot_vec[hot_idx];
                        row[cam_width + 1] = hot_idx as f64;
                    }
                }

                row[cam_width + 2] = tree_id as f64;

                compile_path(&path, &tree.nodes, &mut row);
                cam.append(&mut row);
            }
        }
    }

    cam
}

fn compile_path(path: &[usize], nodes: &[Node], row: &mut [f64]) {
    for (i, node_id) in path.iter().enumerate().take(path.len() - 1) {
        let Node::Internal {
            split_feature_id,
            threshold,
            left_child,
            mut comparison_op,
            ..
        } = &nodes[*node_id]
        else {
            unreachable!()
        };

        if path[i + 1] != *left_child {
            comparison_op = comparison_op.invert();
        }

        match comparison_op {
            ComparisonOp::Greater | ComparisonOp::GreaterEquals => {
                let prev = row[split_feature_id * 2];
                if prev.is_nan() || *threshold > prev {
                    row[split_feature_id * 2] = *threshold;
                }
            }
            ComparisonOp::Less | ComparisonOp::LessEquals => {
                let prev = row[split_feature_id * 2 + 1];
                if prev.is_nan() || *threshold < prev {
                    row[split_feature_id * 2 + 1] = *threshold;
                }
            }
        };
    }
}
