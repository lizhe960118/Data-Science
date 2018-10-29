promos = []

def promotion(promo_func):
    promos.append(promo_func)
    return promos

@promotion
def fidelity(order):
    if order.customer.fidelity >= 1000:
        return order.total() * 0.05
    else:
        return 0

@promotion
def bulk_item(order):
    discount = 0
    for item in order.cart:
        if item.quantity >= 20:
            discount += item.total() * 0.1
    return discount

@promotion
def large_order(order):
    discount_items = {item.produnct for item in order.cart}
    if len(discount_items) >= 10:
        return order.total() * 0.07
    else:
        return 0

def best_promo(order):
    return max(promo(order) for promo in promos)
