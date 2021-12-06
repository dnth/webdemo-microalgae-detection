from icevision_dashboards.data import BboxRecordDataset
from icevision_dashboards.dashboards import ObjectDetectionDatasetOverview
import icedata
# load some data from the icedata
data_dir = icedata.fridge.load_data()
class_map = icedata.fridge.class_map()
parser = icedata.fridge.parser(data_dir)
train_records, valid_records = parser.parse()
# create a dataset that can be consumed by the dashboards
train_dash_ds = BboxRecordDataset(train_records, class_map)
# create a new dashboard instance and display it with the .show() function
overview_dashboard = ObjectDetectionDatasetOverview(train_dash_ds, width=1500, height=900)
overview_dashboard.show()
