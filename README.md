# htt_annotation_example

Example weights and scripts to annotate ntuples for HTT analyses

## How to run it?

Read the script `run_annotation.sh`. It sets up the LCG 92 release from the `sft.cern.ch` CVMFS repository. This example should run out-of-the-box on any lxplus machine.
Use any of your analysis ntuples and run the annotation as follows.

```bash
FILE=/path/to/some/file
TAG=prefix_of_annotated_branches
bash run_annotation.sh FILE TAG
```

The scripts loops through the directories of your input file, searches for names beginning with `mt_` and annotates the tree `ntuples` in this directory.
The annotations can be found in branches of this tree with the set prefix `TAG`.

## Which variables are used?

Check out the file `mt_training_config.yaml`, there are the variables defined. The list in the entry `classes` defines numbers for the output classes of the classifier.

## How to do the categorization of my analysis?

The cut-strings should look like the following. Note that you still have to apply the correct event weights. Here is an example for the VBF category:

```python
cut_string = "TAG_max_index==1"
discriminating_variable = "TAG_max_score"
```
