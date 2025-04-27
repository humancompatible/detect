
def recurse_generate(data, n_min, size, i, lengths, order, offset, pos, neg, counter):
    if sum((data[:, pos] == 1).all(axis=1) & (data[:, neg] == 0).all(axis=1)) < n_min:
        val = recurse(size, i, [lengths[v] for v in order], counter)
        counter.n_checked += val
        counter.n_skipped += val
        return
    if size == 0:
        yield (pos, neg)
        return
    for j in range(i, len(order) - size + 1):
        if lengths[order[j]] == 1:
            yield from recurse_generate(
                data,
                n_min,
                size - 1,
                j + 1,
                lengths,
                order,
                offset + 1,
                pos + [offset],
                neg,
                counter
            )
            yield from recurse_generate(
                data,
                n_min,
                size - 1,
                j + 1,
                lengths,
                order,
                offset + 1,
                pos,
                neg + [offset],
                counter
            )
        else:
            for k in range(lengths[order[j]]):
                yield from recurse_generate(
                    data,
                    n_min,
                    size - 1,
                    j + 1,
                    lengths,
                    order,
                    offset + lengths[order[j]],
                    pos + [offset + k],
                    neg,
                    counter
                )
        offset += lengths[order[j]]


# def generator(data, n_min, feature_order, feature_lens):
# options = list(range(n_options[size]))
# while len(options) > 0:
#     opt_i = np.random.choice(len(options), 1)
#     opt = options[opt_i]
#     options.pop(opt_i)


def recurse(size, i, lengths, counter):
    if size == 0:
        return 1
    tot = 0
    for j in range(i, len(lengths) - size + 1):
        tot += max(lengths[j], 2) * recurse(size - 1, j + 1, lengths, counter)
    return tot


def subg_generator(data, n_min, binarizer, counter, logger):
    feature_order = []
    feature_lens = {}
    for bin in binarizer.get_bin_encodings():
        if bin.feature.name not in feature_lens:
            feature_order.append(bin.feature.name)
            feature_lens[bin.feature.name] = 1
        else:
            feature_lens[bin.feature.name] += 1

    # feature_lens = {k: max(v, 2) for k, v in feature_lens.items()}
    for size in range(1, len(feature_order)):
        counter.n_options += recurse(size, 0, list(feature_lens.values()), counter)
    logger.info(f"In total, there are {counter.n_options} possible subgroups")

    for size in range(1, len(feature_order)):
        yield from recurse_generate(
            data, n_min, size, 0, feature_lens, feature_order, 0, [], [], counter
        )
