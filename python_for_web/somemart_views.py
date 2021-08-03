import json

from django.http import HttpResponse, JsonResponse
from django.views import View
from django.shortcuts import get_object_or_404

from marshmallow import Schema, fields
from marshmallow.validate import Length, Range
from marshmallow import ValidationError

from .models import Item, Review


class AddItemSchema(Schema):
    title = fields.Str(validate=Length(1, 64), required=True)
    description = fields.Str(validate=Length(1, 1024), required=True)
    price = fields.Int(validate=Range(1, 1000000), required=True)


class PostReviewSchema(Schema):
    text = fields.Str(validate=Length(1, 1024), required=True)
    grade = fields.Int(validate=Range(1, 10), required=True)


class AddItemView(View):
    """View для создания товара."""

    def post(self, request):
        try:
            document = json.loads(request.body)
            schema = AddItemSchema(strict=True)
            data = schema.load(document)
            item = Item(title=data.data['title'], description=data.data['description'], price=data.data['price'])
            item.save()
            return JsonResponse({'id': item.id}, status=201)
        except json.JSONDecodeError:
            return JsonResponse({'errors': 'Invalid JSON'}, status=400)
        except ValidationError as exc:
            return JsonResponse({'errors': exc.messages}, status=400)


class PostReviewView(View):
    """View для создания отзыва о товаре."""

    def post(self, request, item_id):
        try:
            document = json.loads(request.body)
            schema = PostReviewSchema(strict=True)
            data = schema.load(document)
            item = get_object_or_404(Item, pk=int(item_id))
            review = Review(grade=data.data['grade'], text=data.data['text'], item=item)
            review.save()
            return JsonResponse({'id': review.id}, status=201)
        except json.JSONDecodeError:
            return JsonResponse({'errors': 'Invalid JSON'}, status=400)
        except ValidationError as exc:
            return JsonResponse({'errors': exc.messages}, status=400)


class GetItemView(View):
    """View для получения информации о товаре.

    Помимо основной информации выдает последние отзывы о товаре, не более 5
    штук.
    """

    def get(self, request, item_id):
        data = {
            'id': '',
            'title': '',
            'description': '',
            'price': '',
            'reviews': []
        }
        item = get_object_or_404(Item, pk=int(item_id))
        data['id'] = item.id
        data['title'] = item.title
        data['description'] = item.description
        data['price'] = item.price
        review = Review.objects.filter(item=item)

        if len(review) <= 5:
            for r in review:
                data['reviews'].append({
                    'id': r.id,
                    'text': r.text,
                    'grade': r.grade
                })
        elif len(review) > 5:
            review = review[::-1][:5]
            for r in review:
                data['reviews'].append({
                    'id': r.id,
                    'text': r.text,
                    'grade': r.grade
                })

        return JsonResponse(data, status=200)




