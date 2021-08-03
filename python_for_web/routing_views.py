from django.http import HttpResponse

from django.views.decorators.http import require_GET, require_http_methods, require_POST


def simple_route(request):
    method = request.method

    if method == 'GET':
        return HttpResponse("", status=200)
    else:
        return HttpResponse("", status=405)


def slug_route(request, first=None, second=None):
    if second is None:
        return HttpResponse(content=first, status=200)
    elif (first != None) and (second != None):
        ans = str(first) + str(second)
        return HttpResponse(content=ans, status=200)
    else:
        return HttpResponse(status=404)


def sum_route(request, first, second):
    a = int(first)
    b = int(second)
    return HttpResponse(a+b)


@require_http_methods(['GET'])
def sum_get_method(request):
    first = request.GET.get('a')
    second = request.GET.get('b')

    if (first == None) or (second == None):
        return HttpResponse(status=400)
    elif check_int(first) and check_int(second):
        return HttpResponse(content=str(int(first) + int(second)), status=200)
    else:
        return HttpResponse(status=400)


@require_http_methods(['POST'])
def sum_post_method(request):
    first = request.POST.get('a')
    second = request.POST.get('b')
    
    if (first == None) or (second == None):
        return HttpResponse(status=400)
    elif check_int(first) and check_int(second):
        return HttpResponse(content=str(int(first) + int(second)), status=200)
    else:
        return HttpResponse(status=400)


def check_int(s):
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()
    