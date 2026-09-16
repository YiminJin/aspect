/* Copyright (C) 2026 by the authors of the ASPECT code.
 * SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef _aspect_reconstructed_fault_sparse_coupling_h
#define _aspect_reconstructed_fault_sparse_coupling_h

#include <aspect/global.h>
#include <map>
#include <vector>

namespace aspect
{
  namespace internal
  {
    /** Rank-local additive piece of a rectangular coupling matrix. Only
     * nonempty rows are stored. Ownership/reduction belongs to B or G. */
    struct FaultSparseCoupling
    {
      using Index = dealii::types::global_dof_index;
      using Entries = std::map<std::pair<Index,Index>,double>;
      std::vector<Index> rows, columns;
      std::vector<std::size_t> offsets;
      std::vector<double> values;

      void build(const Entries &entries)
      {
        for (const auto &entry : entries)
          if (entry.second != 0.)
            {
              if (rows.empty() || rows.back()!=entry.first.first)
                { rows.push_back(entry.first.first); offsets.push_back(values.size()); }
              columns.push_back(entry.first.second);
              values.push_back(entry.second);
            }
        offsets.push_back(values.size());
      }

      template <class Source, class Destination>
      void add(const Source &source, Destination &destination) const
      {
        for (std::size_t r=0; r<rows.size(); ++r)
          {
            double value=0.;
            for (std::size_t j=offsets[r]; j<offsets[r+1]; ++j)
              value+=values[j]*source[columns[j]];
            destination[rows[r]]+=value;
          }
      }

      std::size_t bytes() const
      {
        return sizeof(*this)+(rows.capacity()+columns.capacity())*sizeof(Index)
               +offsets.capacity()*sizeof(std::size_t)+values.capacity()*sizeof(double);
      }
    };
  }
}
#endif
