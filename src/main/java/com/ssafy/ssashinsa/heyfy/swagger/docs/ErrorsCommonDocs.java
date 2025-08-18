package com.ssafy.ssashinsa.heyfy.swagger.docs;

import com.ssafy.ssashinsa.heyfy.common.exception.ErrorResponse;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;

import java.lang.annotation.*;

@Target(ElementType.TYPE) // 클래스에만
@Retention(RetentionPolicy.RUNTIME)
@Documented
@ApiResponses({
        @ApiResponse(
                responseCode = "401",
                description = "인증 실패(401)",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = {@ExampleObject(name = "인증 실패", ref = "#/components/examples/Unauthorized")}
                )
        ),
        @ApiResponse(
                responseCode = "403",
                description = "인가 실패(403)",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = {@ExampleObject(name = "인가 실패", ref = "#/components/examples/Forbidden")}
                )
        )
})
public @interface ErrorsCommonDocs {
}
