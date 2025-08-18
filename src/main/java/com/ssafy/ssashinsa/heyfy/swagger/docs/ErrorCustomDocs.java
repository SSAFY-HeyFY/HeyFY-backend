package com.ssafy.ssashinsa.heyfy.swagger.docs;

import com.ssafy.ssashinsa.heyfy.common.exception.ErrorResponse;
import com.ssafy.ssashinsa.heyfy.swagger.dto.ResultSuccessResponseDto;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;

import java.lang.annotation.*;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Documented
@ApiResponses({
        @ApiResponse(responseCode = "200", description = "메뉴 수정 성공",
                content = @Content(schema = @Schema(implementation = ResultSuccessResponseDto.class))),
        @ApiResponse(responseCode = "400", description = "잘못된 요청",
                content = @Content(mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = {
                                @ExampleObject(name = "테스트",
                                        value = "{\"code\": 400, \"message\": \"테스트\"}"),
                                @ExampleObject(name = "필수 정보 누락",
                                        ref = "#/components/examples/MissingRequired"),
                                @ExampleObject(name = "참가자가아님",
                                        ref = "#/components/examples/NotParticipant")
                        }))
})
public @interface ErrorCustomDocs {
}
