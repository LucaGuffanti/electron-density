#!/bin/bash

filenames=$(ls data/benchmark/*.pdb)

#set the output file to the command line paramter
if [ -n "$1" ]; then
    output="$1"
else
    echo "No output file specified, using default: times.csv"
    output="times.csv"
fi

touch $output
echo "atoms,points,atom,bin,bin_excl" > $output

echo "Running benchmark for the following files:"

for filename in $filenames; do
    echo "  - $filename"
done

echo "---------------------------"

iterations=50

for filename in $filenames; do
    echo "  - $filename"
    number=$(basename $filename | sed 's/.*_\([0-9]*\)\.pdb/\1/')

    echo "Running $iterations iterations for $number atoms"
    filename=$(realpath $filename)

    printf "%-22s %-22s %-22s %-22s %-22s %-22s\n" "" "-----------" "-----------" "-----------" "-----------" "-----------"



    printf "%-22s %-22s %-22s %-22s %-22s %-22s\n" "" "Points" "Atom" "Bin" "Bin_Excl" "Bin_Excl_Opt"
    printf "%-22s %-22s %-22s %-22s %-22s %-22s\n" "" "-----" "-----" "-----" "-----" "-----" 
    point_times=()
    atom_times=()
    bin_times=()
    bin_excl_times=()
    bin_excl_opt_times=()

    touch errors.txt
    echo "" > errors.txt  # Clear the errors file before each run

    for ((i=1; i<=iterations; i++)); do

        # run the all points
        ./build/test/eddi/bench_ALL_POINTS $filename > out 2>> errors.txt
        point_time=$(grep "FINAL KERNEL TIME" out | awk '{print $4}' 2>> errors.txt || echo 0)
        point_times+=($point_time)

        # run the all bins
        ./build/test/eddi/bench_ALL_BINS $filename > out 2>> errors.txt
        bin_time=$(grep "FINAL KERNEL TIME" out | awk '{print $4}' 2>> errors.txt || echo 0)
        bin_times+=($bin_time)

        # run the atom-based
        ./build/test/eddi/bench_ATOM_BASED $filename > out 2>> errors.txt
        atom_time=$(grep "FINAL KERNEL TIME" out | awk '{print $4}' 2>> errors.txt || echo 0)
        atom_times+=($atom_time)

        # run the block-per-bin
        ./build/test/eddi/bench_ONE_BLOCK_PER_BIN $filename > out 2>> errors.txt
        bin_excl_time=$(grep "FINAL KERNEL TIME" out | awk '{print $4}' 2>> errors.txt || echo 0)
        bin_excl_times+=($bin_excl_time)

        # run the block-per-bin with exclusion optimization
        ./build/test/eddi/bench_OPT_ONE_BLOCK_PER_BIN $filename > out 2>> errors.txt
        bin_excl_opt_time=$(grep "FINAL KERNEL TIME" out | awk '{print $4}' 2>> errors.txt || echo 0)
        bin_excl_opt_times+=($bin_excl_opt_time)

        # Calculate running average and standard deviation
        avg_point_time=$(echo "${point_times[@]}" | awk '{sum=0; for (i=1; i<=NF; i++) sum+=$i; print sum/NF}')
        avg_atom_time=$(echo "${atom_times[@]}" | awk '{sum=0; for (i=1; i<=NF; i++) sum+=$i; print sum/NF}')
        avg_bin_time=$(echo "${bin_times[@]}" | awk '{sum=0; for (i=1; i<=NF; i++) sum+=$i; print sum/NF}')
        avg_bin_excl_time=$(echo "${bin_excl_times[@]}" | awk '{sum=0; for (i=1; i<=NF; i++) sum+=$i; print sum/NF}')
        avg_bin_excl_opt_time=$(echo "${bin_excl_opt_times[@]}" | awk '{sum=0; for (i=1; i<=NF; i++) sum+=$i; print sum/NF}')

        stddev_point_time=$(echo "${point_times[@]}" | awk -v avg=$avg_point_time '{sum=0; for (i=1; i<=NF; i++) sum+=($i-avg)^2; print sqrt(sum/NF)}')
        stddev_atom_time=$(echo "${atom_times[@]}" | awk -v avg=$avg_atom_time '{sum=0; for (i=1; i<=NF; i++) sum+=($i-avg)^2; print sqrt(sum/NF)}')
        stddev_bin_time=$(echo "${bin_times[@]}" | awk -v avg=$avg_bin_time '{sum=0; for (i=1; i<=NF; i++) sum+=($i-avg)^2; print sqrt(sum/NF)}')
        stddev_bin_excl_time=$(echo "${bin_excl_times[@]}" | awk -v avg=$avg_bin_excl_time '{sum=0; for (i=1; i<=NF; i++) sum+=($i-avg)^2; print sqrt(sum/NF)}')
        stddev_bin_excl_opt_time=$(echo "${bin_excl_opt_times[@]}" | awk -v avg=$avg_bin_excl_opt_time '{sum=0; for (i=1; i<=NF; i++) sum+=($i-avg)^2; print sqrt(sum/NF)}')
        echo "$number,$point_time,$atom_time,$bin_time,$bin_excl_time,$bin_excl_opt_time" >> $output
        # Print results in a table format

        printf "\033[1A\033[K" # Move up one line and clear it

        printf "%-22s %-22s %-22s %-22s %-22s %-22s\n" "$i" "$point_time" "$atom_time" "$bin_time" "$bin_excl_time" "$bin_excl_opt_time"

    
        printf "%-22s %-22s %-22s %-22s %-22s %-22s\n" "Average (std)"\
        "$avg_point_time ($stddev_point_time)" \
        "$avg_atom_time ($stddev_atom_time)" \
        "$avg_bin_time ($stddev_bin_time)" \
        "$avg_bin_excl_time ($stddev_bin_excl_time)" \
        "$avg_bin_excl_opt_time ($stddev_bin_excl_opt_time)"
    done
    printf "\n\n"
done
