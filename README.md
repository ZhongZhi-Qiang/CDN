# CDN
This repo is the Pytorch implementation of our submitted paper: Non-Stationary Time Series Forecasting Considering Cascaded Decomposable Normalization

### Environment

```bash
mkdir datasets
pip install -r requirements.txt
```
All the 8 datasets are available at the [Google Driver](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link) provided by iTransformer. Many thanks to their efforts and devotion!

#### Running

```bash
bash scripts/long_term_forecast/run_<model_name>.sh 
bash scripts/short_term_forecast/<model_name>/run_<model_name>_short.sh 
```