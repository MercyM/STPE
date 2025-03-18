from .basic_controller import BasicMAC
from .cqmix_controller import CQMixMAC
from .EA_basic_controller import RL_BasicMAC, Gen_BasicMAC, Gen_BasicMAC_LTSCG,Gen_BasicMAC_DICG
REGISTRY = {}
REGISTRY["basic_mac"] = BasicMAC
REGISTRY["cqmix_mac"] = CQMixMAC

REGISTRY["RL_basic_mac"] = RL_BasicMAC
REGISTRY["EA_basic_mac"] = Gen_BasicMAC
REGISTRY["EA_basic_mac_ltscg"] = Gen_BasicMAC_LTSCG
REGISTRY["EA_basic_mac_dicg"] = Gen_BasicMAC_DICG