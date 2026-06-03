#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility wrapper for structure-prior extraction.

This keeps the old import path intact while reusing the online prior code that is
now integrated into the model.
"""

from basicsr.models.archs.prior_utils import CIConv2d, MultiInvariantPrior

__all__ = ['CIConv2d', 'MultiInvariantPrior']
