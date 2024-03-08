# EOVSA Flarelist Operations (eovsa-flarelist-ops)

This repository contains scripts essential for updating the MySQL database of the [EOVSA Flarelist website](http://www.ovsa.njit.edu/flarelist) and generating spectrogram data and figures. It serves as a key component in maintaining and populating the flarelist with recent solar flare observations by EOVSA. `eovsa-flarelist-ops` is a submodule of the [EOVSA Flarelist main repository](https://github.com/ovro-eovsa/eovsa-flarelist).


### Prerequisites

Ensure you have Python installed along with necessary dependencies. While the specific dependencies may vary based on your environment and the scripts' requirements, a general setup might include MySQL connectors and data processing libraries such as NumPy and Matplotlib.

### Usage

Before executing `run_flarelist2sql.sh`, switch to the `user` group to ensure proper file access permissions:

```bash
newgrp user
```

Then, run the script with optional arguments:

```bash
./run_flarelist2sql.sh -t "YYYY-MM-DD HH:MM:SS" "YYYY-MM-DD HH:MM:SS"
```

- The `-t` or `--timerange` argument specifies the time range for fetching and processing flare data.
- `--do_manu` (if applicable) enables manual determination of start/end times for radio bursts.

## License

This project is licensed under the [MIT License](LICENSE.md) - see the LICENSE file for details.

