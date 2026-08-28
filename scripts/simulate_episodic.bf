// HyPhy HBL Batch Script: simulate_episodic.bf
// Simulates a dual-partition codon alignment (Null Partition + Episodic Selection Partition)

RequireVersion ("2.5.0");

// Read arguments passed via command line
// Usage: hyphy simulate_episodic.bf <tree_file> <omega_pos> <num_null_sites> <num_alt_sites> <output_nex>

tree_file     = io.PromptUserForFilePath ("Path to tree file");
omega_pos     = io.PromptUserForString ("Foreground omega_pos") + 0.0;
num_null      = io.PromptUserForString ("Number of null sites") + 0;
num_alt       = io.PromptUserForString ("Number of alt sites") + 0;
out_nex_path  = io.PromptUserForFilePath ("Output NEXUS path");

// Read tree topology
tree_raw = String (fscanf (tree_file, "Everything"));

SetDialogPrompt ("Select genetic code");
ExecuteAFile (HYPHY_LIB_DIRECTORY + "TemplateBatchFiles" + DIRECTORY_SEPARATOR + "Modules" + DIRECTORY_SEPARATOR + "GeneticCodes" + DIRECTORY_SEPARATOR + "Universal.gencode");

// 1. Model for Null Sites: alpha = 1.0, beta = 1.0 everywhere
alpha = 1.0;
beta = 1.0;
ExecuteAFile (HYPHY_LIB_DIRECTORY + "TemplateBatchFiles" + DIRECTORY_SEPARATOR + "modules" + DIRECTORY_SEPARATOR + "models" + DIRECTORY_SEPARATOR + "codon" + DIRECTORY_SEPARATOR + "MG94.bf");

Tree tree_null = tree_raw;

// Simulate Null Partition
DataSet sim_null = Simulate (tree_null, Vector (61, 1/61), num_null, 0);

// 2. Model for Alt Sites: beta = 0.2 on background branches, beta = omega_pos * alpha on foreground branches
Tree tree_alt = tree_raw;

// Tag 20% of branches randomly as foreground
branch_names = BranchName (tree_alt, -1);
num_branches = Columns (branch_names);
num_fg = Max (1, Floor (num_branches * 0.20));

// Assign background beta=0.2 and foreground beta=omega_pos
for (b = 0; b < num_branches; b = b + 1) {
    bname = branch_names[b];
    ExecuteCommands ("tree_alt." + bname + ".beta = 0.2;");
}

for (b = 0; b < num_fg; b = b + 1) {
    bname = branch_names[b];
    ExecuteCommands ("tree_alt." + bname + ".beta = " + omega_pos + ";");
}

DataSet sim_alt = Simulate (tree_alt, Vector (61, 1/61), num_alt, 0);

// Concatenate Partitions into a single dataset
DataSet sim_combined = Combine (sim_null, sim_alt);
DataSetFilter filter_all = CreateFilter (sim_combined, 1);

// Write to NEXUS file
fprintf (out_nex_path, CLEAR_FILE, filter_all);
fprintf (out_nex_path, "\nBEGIN TREES;\n  TREE tree = ", tree_raw, "\nEND;\n");
