#pragma once
/*
 *  Copyright (C) 2024  Brett Terpstra
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef COSC_4P80_ASSIGNMENT_3_FILE_H
#define COSC_4P80_ASSIGNMENT_3_FILE_H

#include <blt/std/types.h>
#include <vector>
#include <string>
#include <string_view>
#include "blt/std/assert.h"
#include <assign3/fwdecl.h>

namespace assign3
{
    
    struct data_t
    {
        bool is_bad = false;
        std::vector<Scalar> bins;
        
        [[nodiscard]] data_t normalize() const;
        
        [[nodiscard]] data_t with_padding(blt::size_t desired_size, Scalar padding_value = 0) const;
    };
    
    struct data_file_t
    {
        public:
            std::vector<data_t> data_points;
            
            [[nodiscard]] data_file_t normalize() const;
            
            [[nodiscard]] data_file_t with_padding(blt::size_t desired_size, Scalar padding_value = 0) const;
            
            data_file_t& operator+=(const data_file_t& o);
            
            data_file_t friend operator+(const data_file_t& a, const data_file_t& b);
            
            static std::vector<data_file_t> load_data_files_from_path(std::string_view path);
        
        private:
            static std::vector<std::string> get_data_file_list(std::string_view path);
            
            static std::vector<data_file_t> load_data_files(const std::vector<std::string>& files);
    };
    
    struct partitioned_dataset_t
    {
        public:
            explicit partitioned_dataset_t(std::vector<data_file_t> groups):
                    groups(std::move(groups)), bins(this->groups.begin()->data_points.begin()->bins.size())
            {}
            
            [[nodiscard]] const std::vector<data_file_t>& getGroups() const
            {
                return groups;
            }
            
            [[nodiscard]] blt::size_t bin_size() const
            {
                return bins;
            }
        
        private:
            std::vector<data_file_t> groups;
            blt::size_t bins;
    };
    
    struct dataset_partitioner
    {
        public:
            explicit dataset_partitioner(const data_file_t& file)
            {
                with(file);
            }
            
            dataset_partitioner& with(const data_file_t& data)
            {
                BLT_ASSERT(data.data_points.begin()->bins.size() == files.begin()->data_points.begin()->bins.size());
                files.push_back(data);
                return *this;
            }
            
            [[nodiscard]] partitioned_dataset_t partition(blt::size_t groups) const;
        
        private:
            std::vector<data_file_t> files;
    };
    
    void save_as_csv(const std::string& file, const std::vector<std::pair<std::string, std::vector<Scalar>>>& data);
}

#endif //COSC_4P80_ASSIGNMENT_3_FILE_H
