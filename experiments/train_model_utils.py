"""
Utilities.
"""
import argparse, os, sys
sys.path.append('..')
from src.model_zoo import deep_cnn, shallow_cnn

# ====================
# define constants

BASERESULTSDIR = os.path.abspath("../results")
SYNTHETIC_DATADIR = os.path.abspath("../data")
models = {'shallow':shallow_cnn, 'deep':deep_cnn}
factors = [1, 4, 8]

# =====================

def _validate_args(args):
    assert int(args.factor) in factors
    assert args.type.lower() in models.keys()
    assert isinstance(args.batch, int) and args.batch > 0
    assert isinstance(args.epochs, int) and args.epochs > 0
    assert isinstance(args.start_trial, int) and args.start_trial > 0
    assert isinstance(args.end_trial, int) and args.end_trial > 0
    assert args.end_trial >= args.start_trial
    assert isinstance(args.bn, bool)
    assert isinstance(args.dropout1, float) and args.dropout1 > 0. and args.dropout1 < 1.
    assert isinstance(args.dropout2, float) and args.dropout2 > 0. and args.dropout2 < 1.
    assert isinstance(args.es, bool)
    assert isinstance(args.reduce_lr, bool)
    assert isinstance(args.lr, float) and args.lr > 0.
    assert isinstance(args.track_saliency, bool)
    assert isinstance(args.track_sg, bool)
    assert isinstance(args.track_intgrad, bool)
    assert isinstance(args.activation, str)

def get_keyboard_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default=32, help="Batch size", type=int)
    parser.add_argument("--epochs", default=100, help="Epochs", type=int)
    parser.add_argument("--factor", default=1, help='Expansion factor', type=int)
    parser.add_argument("--type", default='deep', help='Deep or shallow model', type=str)
    parser.add_argument("--start_trial", default=1, help='Index of 1st trial', type=int)
    parser.add_argument("--end_trial", default=10, help="Index of final trial", type=int)
    parser.add_argument("--bn", action="store_true", help="Whether to use batch norm",)
    parser.add_argument("--dropout1", default=0.2, help='Dropout applied to conv layers', type=float)
    parser.add_argument("--dropout2", default=0.5, help="Dropout applied to dense layers", type=float)
    parser.add_argument("--es", action='store_true', help='Whether to add Early Stopping callback',)
    parser.add_argument("--lr", default=1e-3, help='Base learning rate', type=float)
    parser.add_argument("--reduce_lr", action='store_true',
                help='Whether to add lr reduction on plateau callback',)
    parser.add_argument("--track_saliency", action='store_true',
                        help='Whether to track saliency maps during training',)
    parser.add_argument("--track_sg", action='store_true',
                        help='Whether to track smoothgrad during training',)
    parser.add_argument("--track_intgrad", action='store_true',
                        help='Whether to track integrated gradients during training',)
    parser.add_argument("--activation", default='relu', help='First layer activation.', type=str)

    args = parser.parse_args()
    _validate_args(args)
    return args
